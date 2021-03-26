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
Code related to wrapping around API calls under api.neuralmagic.com/[object]/get
"""

import logging
from typing import Dict, Union

import requests

from sparsezoo.requests.authentication import get_auth_header
from sparsezoo.requests.base import MODELS_API_URL, RECIPES_API_URL, ModelArgs


__all__ = ["get_request", "get_model_get_request", "get_recipe_get_request", "GET_PATH"]


_LOGGER = logging.getLogger(__name__)
GET_PATH = "get"


def get_request(
    base_url: str,
    args: Union[ModelArgs, str],
    sub_path: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Get an object from the sparsezoo for any objects matching the args.

    The path called has structure:
        [base_url]/get/[args.stub]/{sub_path}

    :param base_url: the base url of the request
    :param args: the args describing what should be retrieved
    :param file_name: the sub path from the model path if any e.g.
        file_name for models api or recipe_type for the recipes api
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    header = get_auth_header(force_token_refresh=force_token_refresh)
    path = args if isinstance(args, str) else args.stub
    url = f"{base_url}/{GET_PATH}/{path}"

    if sub_path:
        url = f"{url}/{sub_path}"

    if hasattr(args, "release_version") and args.release_version:
        url = f"{url}?release_version={args.release_version}"

    _LOGGER.debug(f"GET download from {url}")

    response = requests.get(url=url, headers=header)
    response.raise_for_status()
    response_json = response.json()

    return response_json


def get_model_get_request(
    args: Union[ModelArgs, str],
    file_name: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Get a model from the sparsezoo for any objects matching the args

    :param args: the model args describing what should be retrieved for
    :param file_name: the name of the file, if any, to get model info for
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    return get_request(
        MODELS_API_URL,
        args=args,
        sub_path=file_name,
        force_token_refresh=force_token_refresh,
    )


def get_recipe_get_request(
    args: Union[ModelArgs, str],
    recipe_type: Union[str, None] = None,
    force_token_refresh: bool = False,
):
    """
    Get a recipe from the sparsezoo for any objects matching the args

    :param args: the model args describing what should be retrieved for
    :param recipe_type: the recipe_type to get recipe info for if not original
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    return get_request(
        base_url=RECIPES_API_URL,
        args=args,
        sub_path=recipe_type,
        force_token_refresh=force_token_refresh,
    )
