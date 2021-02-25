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
Code related to wrapping around API calls under api.neuralmagic.com/objects/search
"""

import logging
from typing import Dict

import requests

from sparsezoo.requests.authentication import get_auth_header
from sparsezoo.requests.base import BASE_API_URL, ModelArgs


__all__ = ["search_get_request", "SEARCH_PATH"]


_LOGGER = logging.getLogger(__name__)
SEARCH_PATH = "search"


def search_get_request(
    args: ModelArgs,
    page: int = 1,
    page_length: int = 20,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Search the sparsezoo for any objects matching the args

    :param args: the model args describing what should be searched for
    :param page: the page of values to get
    :param page_length: the page length of values to get
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    if not page > 0:
        raise Exception("'page' value must be > 0")

    if not page_length > 0:
        raise Exception("'page_length' value must be > 0")

    header = get_auth_header(force_token_refresh=force_token_refresh)

    search_args = args.model_url_args
    search_args.extend([f"page={page}", f"page_length={page_length}"])

    if args.release_version:
        search_args.extend(f"release_version={args.release_version}")

    search_args = "&".join(search_args)
    url = f"{BASE_API_URL}/{SEARCH_PATH}/{args.model_url_root}?{search_args}"

    _LOGGER.info(f"Searching objects from {url}")
    response_json = requests.get(url=url, headers=header).json()

    return response_json
