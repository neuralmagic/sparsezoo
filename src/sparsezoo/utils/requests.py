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
from typing import Dict, Union

import requests

from sparsezoo.utils.authentication import get_auth_header


BASE_API_URL = (
    os.getenv("SPARSEZOO_API_URL")
    if os.getenv("SPARSEZOO_API_URL")
    else "https://api.neuralmagic.com"
)
MODELS_API_URL = f"{BASE_API_URL}/models"


__all__ = ["download_get_request"]

_LOGGER = logging.getLogger(__name__)

DOWNLOAD_PATH = "download"


def download_get_request(
    args: str,
    base_url: str = MODELS_API_URL,
    sub_path: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Get a downloadable object from the sparsezoo for any objects matching the args

    The path called has structure:
        [base_url]/download/[args.stub]/{sub_path}

    :param base_url: the base url
    :param args: the model args describing what should be downloaded for
    :param sub_path: the sub path from the model path if any e.g.
        file_name for models api or recipe_type for the recipes api
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    header = get_auth_header(force_token_refresh=force_token_refresh)
    path = args if isinstance(args, str) else args.stub
    url = f"{base_url}/{DOWNLOAD_PATH}/{path}"

    if sub_path:
        url = f"{url}/{sub_path}"

    download_args = []

    if hasattr(args, "release_version") and args.release_version:
        download_args.append(f"release_version={args.release_version}")

    if download_args:
        url = f"{url}?{'&'.join(download_args)}"

    _LOGGER.debug(f"GET download from {url}")
    response = requests.get(url=url, headers=header)

    response.raise_for_status()
    response_json = response.json()

    return response_json
