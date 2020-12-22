import os
import logging

import requests

from sparsezoo.schemas import FileSchema, ModelSchema
from sparsezoo.utils import get_auth_header, BASE_API_URL


__all__ = ["download_model_file", "download_model"]


_LOGGER = logging.getLogger(__name__)


def download_model(
    domain: str,
    sub_domain: str,
    architecture: str,
    sub_architecture: str,
    dataset: str,
    framework: str,
    optimization_name: str,
    release_version: str = None,
    overwrite: bool = False,
    save_dir: str = None,
    save_path: str = None,
    download_files: bool = True,
    refresh_token: bool = False,
) -> ModelSchema:
    header = get_auth_header(refresh_token=refresh_token)

    url = os.path.join(
        BASE_API_URL,
        "download",
        domain,
        sub_domain,
        architecture,
        sub_architecture,
        dataset,
        framework,
        optimization_name,
    )
    if release_version:
        url += f"?release_version={release_version}"
    _LOGGER.info(f"Obtaining model from {url}")

    response = requests.get(url=url, headers=header)
    response.raise_for_status()
    response_json = response.json()

    model = ModelSchema(**response_json["model"])
    if download_files:
        save_location = model.download(
            overwrite=overwrite, save_dir=save_dir, save_path=save_path
        )
        _LOGGER.info(f"Successfully saved model files at {save_location}")
    return model


def download_model_file(
    domain: str,
    sub_domain: str,
    architecture: str,
    sub_architecture: str,
    dataset: str,
    framework: str,
    optimization_name: str,
    file_name: str,
    release_version: str = None,
    overwrite: bool = False,
    save_dir: str = None,
    save_path: str = None,
    refresh_token: bool = False,
) -> FileSchema:
    header = get_auth_header(refresh_token=refresh_token)

    url = os.path.join(
        BASE_API_URL,
        "download",
        domain,
        sub_domain,
        architecture,
        sub_architecture,
        dataset,
        framework,
        optimization_name,
        file_name,
    )
    _LOGGER.info(f"Obtaining model file at {url}")

    if release_version:
        url += f"?release_version={release_version}"

    response = requests.get(url=url, headers=header)
    response.raise_for_status()
    response_json = response.json()

    file_obj = FileSchema(**response_json["file"])

    save_location = file_obj.download(
        overwrite=overwrite, save_dir=save_dir, save_path=save_path
    )
    _LOGGER.info(f"Successfully saved file at {save_location}")
    return file_obj
