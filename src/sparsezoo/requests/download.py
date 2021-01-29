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
Code related to wrapping around API calls under api.neuralmagic.com/objects/download
"""

import logging
from typing import Dict, Union

import requests

from sparsezoo.requests.authentication import get_auth_header
from sparsezoo.requests.base import BASE_API_URL, ModelArgs


__all__ = ["download_get_request", "DOWNLOAD_PATH"]


_LOGGER = logging.getLogger(__name__)
DOWNLOAD_PATH = "download"


def download_get_request(
    args: Union[ModelArgs, str],
    file_name: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Get a downloadable model from the sparsezoo for any objects matching the args

    :param args: the model args describing what should be downloaded for
    :param file_name: the name of the file, if any, to get download info for
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    header = get_auth_header(force_token_refresh=force_token_refresh)
    path = args if isinstance(args, str) else args.stub
    url = f"{BASE_API_URL}/{DOWNLOAD_PATH}/{path}"

    if file_name:
        url = f"{url}/{file_name}"

    if hasattr(args, "release_version") and args.release_version:
        url = f"{url}?release_version={args.release_version}"

    _LOGGER.debug(f"GET download from {url}")

    response = requests.get(url=url, headers=header)
    response.raise_for_status()
    response_json = response.json()

    return response_json
