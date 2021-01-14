"""
Code related to wrapping around API calls under api.neuralmagic.com/objects/search
"""

from typing import Dict
import logging
import requests

from sparsezoo.requests.base import BASE_API_URL, ModelArgs
from sparsezoo.requests.authentication import get_auth_header


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

    _LOGGER.debug(f"Searching objects from {url}")
    response_json = requests.get(url=url, headers=header).json()

    return response_json
