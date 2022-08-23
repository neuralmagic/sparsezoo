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

import copy
import logging
import os
import urllib
from typing import Dict, Union

import requests

from sparsezoo.utils import MODELS_API_URL, convert_to_bool
from sparsezoo.utils.authentication import get_auth_header


__all__ = ["download_get_request", "search_model_get_request"]

_LOGGER = logging.getLogger(__name__)

DOWNLOAD_PATH = "download"
SEARCH_PATH = "search"


def search_get_request(
    base_url: str,
    args: Dict[str, str],
    page: int = 1,
    page_length: int = 20,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Search the sparsezoo for any objects matching the args
    :param base_url: the base url
    :param args: the dictionary describing what should be searched for
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

    search_args = copy.copy(args)
    search_args.update({"page": page, "page_length": page_length})

    if "release_version" in args:
        search_args.update({"release_version": args["release_version"]})

    model_url_root = args["domain"]
    if "sub_domain" in args:
        model_url_root += f"/{args['sub_domain']}"

    search_args = urllib.parse.urlencode(search_args)

    url = f"{base_url}/{SEARCH_PATH}/{model_url_root}?{search_args}"

    _LOGGER.info(f"Searching objects from {url}")
    response_json = requests.get(url=url, headers=header).json()

    return response_json


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

    if convert_to_bool(os.getenv("SPARSEZOO_TEST_MODE")):
        # important, do not remove
        download_args.append("increment_download=False")

    if download_args:
        url = f"{url}?{'&'.join(download_args)}"

    _LOGGER.debug(f"GET download from {url}")
    response = requests.get(url=url, headers=header)

    response.raise_for_status()
    response_json = response.json()
    return response_json


def search_model_get_request(
    args: Dict[str, str],
    page: int = 1,
    page_length: int = 20,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Search the sparsezoo for any models matching the args
    :param args: the dictionary describing what should be searched for
    :param page: the page of values to get
    :param page_length: the page length of values to get
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    return search_get_request(
        base_url=MODELS_API_URL,
        args=args,
        page=page,
        page_length=page_length,
        force_token_refresh=force_token_refresh,
    )
