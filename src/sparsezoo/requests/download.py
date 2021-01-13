from typing import Union, Dict
import logging
import requests

from sparsezoo.requests.base import BASE_API_URL, ModelArgs
from sparsezoo.requests.authentication import get_auth_header


__all__ = ["download_get_request", "DOWNLOAD_PATH"]


_LOGGER = logging.getLogger(__name__)
DOWNLOAD_PATH = "download"


def download_get_request(
    args: ModelArgs,
    file_name: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Dict:
    """
    [TODO]
    """
    header = get_auth_header(force_token_refresh=force_token_refresh)
    arch_id = f"{architecture}-{sub_architecture}" if sub_architecture else architecture
    training_id = f"{dataset}-{training_scheme}" if training_scheme else dataset
    optimization_id = (
        f"{optim_name}-{optim_category}-{optim_target}"
        if optim_target
        else f"{optim_name}-{optim_category}"
    )

    url = "/".join(
        [
            BASE_API_URL,
            DOWNLOAD_PATH,
            domain,
            sub_domain,
            arch_id,
            framework,
            repo,
            training_id,
            optimization_id,
        ]
    )

    if file_name:
        url += f"/{file_name}"

    if release_version:
        url += f"?release_version={release_version}"

    _LOGGER.debug(f"GET download from {url}")

    response = requests.get(url=url, headers=header)
    response.raise_for_status()
    response_json = response.json()

    return response_json
