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
from typing import Dict, List, Optional, Tuple, Union

from sparsezoo.v2.requests.requests import download_get_request


ZOO_STUB_PREFIX = "zoo:"

__all__ = ["load_files_from_stub"]

_LOGGER = logging.getLogger(__name__)
BASE_API_URL = (
    os.getenv("SPARSEZOO_API_URL")
    if os.getenv("SPARSEZOO_API_URL")
    else "https://api.neuralmagic.com"
)
MODELS_API_URL = f"{BASE_API_URL}/models"


def filter_files(files, params):
    ((param, value),) = params.items()
    if param == "recipe":
        files_filtered = [
            file_dict
            for file_dict in files
            if file_dict["file_type"] == param
            and file_dict["display_name"] == "recipe_" + value + ".md"
        ]
    elif param == "deployment":
        files_filtered = [
            file_dict for file_dict in files if file_dict["file_type"] == param
        ]
    else:
        files_filtered = [
            file_dict for file_dict in files if file_dict["file_type"] == "training"
        ]

    if not files_filtered:
        raise ValueError("")
    else:
        return files_filtered


def file_dictionary(**kwargs):
    return kwargs


def load_files_from_directory(directory_path):
    files = []
    display_names = os.listdir(directory_path)
    if not display_names:
        raise ValueError(
            "The directory path is empty. "
            "Check whether the indicated directory exists."
        )
    for display_name in display_names:
        files.append(
            file_dictionary(
                display_name=display_name,
                path=os.path.join(directory_path, display_name),
            )
        )
    return files


def load_files_from_stub(
    stub: str,
    force_token_refresh: bool = False,
) -> "Model":
    """
    :param stub: the SparseZoo stub path to the model (string path)
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: The requested Model instance
    """
    if isinstance(stub, str):
        stub, params = _parse_zoo_stub(stub, valid_params=[])
    _LOGGER.debug(f"load_model_from_stub: loading model from {stub}")
    response_json = download_get_request(
        base_url=MODELS_API_URL,
        args=stub,
        # sub_path=file_name,
        force_token_refresh=force_token_refresh,
    )
    return response_json["model"]["files"], params


def _parse_zoo_stub(
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
