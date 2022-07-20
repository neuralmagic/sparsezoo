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

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

from sparsezoo.v2.requests.requests import download_get_request
from sparsezoo.v2.utils.backwards_compatibility import restructure_request_json


__all__ = ["load_files_from_stub", "load_files_from_directory"]

_LOGGER = logging.getLogger(__name__)

ZOO_STUB_PREFIX = "zoo:"

BASE_API_URL = (
    os.getenv("SPARSEZOO_API_URL")
    if os.getenv("SPARSEZOO_API_URL")
    else "https://api.neuralmagic.com"
)
MODELS_API_URL = f"{BASE_API_URL}/models"


def file_dictionary(**kwargs):
    return kwargs


def load_files_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    :param directory_path: a path to the directory,
        that contains model files in the expected structure
    :return list of file dictionaries
    """
    display_names = os.listdir(directory_path)
    if not display_names:
        raise ValueError(
            "The directory path is empty. "
            "Check whether the indicated directory exists."
        )
    files = [
        file_dictionary(
            display_name=display_name, path=os.path.join(directory_path, display_name)
        )
        for display_name in display_names
    ]
    return files


def filter_files(
    files: List[Dict[str, Any]], params: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Use the `params` to extract only the relevant files from `files`

    :param files: a list of file dictionaries
    :param params: a dictionary with filtering parameters
    :return a filtered `files` object
    """
    ((param, value),) = params.items()
    if param == "recipe":
        # pick a recipe-type files with the correct file name
        files_filtered = [
            file_dict
            for file_dict in files
            if file_dict["file_type"] == param
            and file_dict["display_name"] == "recipe_" + value + ".md"
        ]
    elif param == "deployment":
        # pick deployment-type files
        files_filtered = [
            file_dict for file_dict in files if file_dict["file_type"] == param
        ]
    else:
        # pick training-type files
        files_filtered = [
            file_dict for file_dict in files if file_dict["file_type"] == "training"
        ]
    if not files_filtered:
        raise ValueError("No files found! The list of files is empty!")
    else:
        return files_filtered


def load_files_from_stub(
    stub: str,
    valid_params: Optional[List[str]] = None,
    force_token_refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    :param stub: the SparseZoo stub path to the model (optionally
        may include string arguments)
    :param valid_params: list of expected parameter names to be encoded in the
        stub. Will raise a warning if any unexpected param names are given. Leave
        as None to not raise any warnings. Default is None
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: The tuple of
        - list of file dictionaries
        - parsed param dictionary
    """
    if isinstance(stub, str):
        stub, params = parse_zoo_stub(stub, valid_params=valid_params)
    _LOGGER.debug(f"load_model_from_stub: loading model from {stub}")
    response_json = download_get_request(
        base_url=MODELS_API_URL,
        args=stub,
        force_token_refresh=force_token_refresh,
    )
    # piece of code required for backwards compatibility
    files = restructure_request_json(response_json["model"]["files"])
    if params:
        files = filter_files(files, params)
    else:
        files = response_json["model"]["files"]
    return files, params


def parse_zoo_stub(
    stub: str, valid_params: Optional[List[str]] = None
) -> Tuple[str, Dict[str, str]]:
    """
    :param stub: A SparseZoo model stub. i.e. 'model/stub/path',
        'zoo:model/stub/path', 'zoo:model/stub/path?param1=value1&param2=value2'
    :param valid_params: list of expected parameter names to be encoded in the
        stub. Will raise a warning if any unexpected param names are given. Leave
        as None to not raise any warnings. Default is None
    :return: the parsed base stub and a dictionary of parameter names and their values
    """
    # strip optional zoo stub prefix
    if stub.startswith(ZOO_STUB_PREFIX):
        stub = stub[len(ZOO_STUB_PREFIX) :]

    if "?" not in stub:
        return stub, {}

    stub_parts = stub.split("?")
    if len(stub_parts) > 2:
        raise ValueError(
            "Invalid SparseZoo stub, query string must be preceded by only one '?'"
            f"given {stub}"
        )
    stub, params = stub_parts
    params = dict(param.split("=") for param in params.split("&"))

    if valid_params is not None and any(param not in valid_params for param in params):
        warnings.warn(
            f"Invalid query string for stub {stub} valid params include {valid_params},"
            f" given {list(params.keys())}"
        )

    return stub, params
