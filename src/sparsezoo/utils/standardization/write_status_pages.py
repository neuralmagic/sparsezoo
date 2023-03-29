# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper functions to sweep directories and write status page markdown files from
yaml configs
"""


import logging
import os
from typing import List, Optional, Type

from sparsezoo.utils.standardization import FeatureStatusPage


__all__ = [
    "write_status_pages",
]


_LOGGER = logging.getLogger(__name__)


def write_status_pages(
    status_page_class: Type[FeatureStatusPage],
    root_directory: str,
    main_status_page_path: Optional[str] = None,
    yaml_template_path: Optional[str] = None,
    status_page_yaml_extension: str = ".status.yaml",
    repo_name: str = "",
):
    """
    Recursively sweeps the root directory for status page yaml files of the given class
    and generates a markdown version of the status page with the same file name
    but extension `.status.md`.

    If main_status_page_path is provided, an aggregated markdown status page of
    all found yaml files will be written there.

    If yaml_template_path is provided a template yaml config of the status page
    will be written there.
    Writes markdown generated status pages

    :param status_page_class: target FeatureStatusPage class to load configs into
    :param root_directory: rood directory to recursively search for yaml configs from
    :param main_status_page_path: if provided will write an aggregated markdown status
        page of all found yaml files to this path
    :param yaml_template_path: if provided will write a yaml template of this status
        page class to this path
    :param status_page_yaml_extension: yaml extension to match when searching for
        configs. Default is '.status.yaml'
    :param repo_name: optional repo name for merged status page title
    """
    page_class_name = status_page_class.__class__.__name__
    matched_file_paths = _get_target_file_paths(
        root_directory, status_page_yaml_extension
    )

    status_pages = []
    status_page_paths = []
    for yaml_file_path in matched_file_paths:
        try:
            status_page = status_page_class.from_yaml(yaml_file_path)
            status_pages.append(status_page)
            status_page_paths.append(yaml_file_path)
        except Exception:
            _LOGGER.warning(f"Unable to load {yaml_file_path} into {page_class_name}")

    for yaml_path, status_page in zip(status_page_paths, status_pages):
        markdown_path = yaml_path.replace(status_page_yaml_extension, ".status.md")
        _LOGGER.info(f"Writing status page: {markdown_path} from {yaml_path}")
        _write_str_to_file(status_page.markdown(), markdown_path)

    if main_status_page_path:
        _LOGGER.info(f"Writing merged status page to {main_status_page_path}")
        merged_status_page_str = status_page_class.merged_markdown(
            status_pages, repo_name=repo_name
        )
        _write_str_to_file(merged_status_page_str, main_status_page_path)

    if yaml_template_path:
        _LOGGER.info(f"Writing template for {page_class_name} to {yaml_template_path}")
        _write_str_to_file(status_page_class.template_yaml_str(), yaml_template_path)


def _get_target_file_paths(root_dir: str, extension: str) -> List[str]:
    target_files = set()
    for directory, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                target_files.add(os.path.join(directory, file))
    return list(target_files)


def _write_str_to_file(content: str, path: str):
    with open(path, "w") as file:
        file.write(content)
