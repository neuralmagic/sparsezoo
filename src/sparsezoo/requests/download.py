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
    args: ModelArgs,
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
    url = f"{BASE_API_URL}/{DOWNLOAD_PATH}/{args.model_url_path}"

    if file_name:
        url = f"{url}/{file_name}"

    if args.release_version:
        url = f"{url}?release_version={args.release_version}"

    _LOGGER.debug(f"GET download from {url}")

    response = requests.get(url=url, headers=header)
    response.raise_for_status()
    response_json = response.json()

    return response_json
