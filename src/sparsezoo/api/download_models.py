"""
Code related to wrapping around API calls under api.neuralmagic.com/models/download
"""

import os
import logging

import requests

from sparsezoo.schemas import RepoFile, RepoModel
from sparsezoo.utils import get_auth_header, BASE_API_URL


__all__ = ["download_model_file", "download_model", "convert_model_to_downloadable"]


_LOGGER = logging.getLogger(__name__)


def convert_model_to_downloadable(
    model: RepoModel,
    refresh_token: bool = False,
) -> RepoModel:
    """
    Converts a model with unsigned files to a model with signed files

    :return: a new version of the model with signed files
    """
    header = get_auth_header(refresh_token=refresh_token)

    url = os.path.join(
        BASE_API_URL,
        "download",
        model.domain,
        model.sub_domain,
        model.architecture,
        model.sub_architecture,
        model.dataset,
        model.framework,
        model.optimization_name,
    )
    url += f"?release_version={str(model.release_version)}"
    _LOGGER.info(f"Obtaining model from {url}")

    response = requests.get(url=url, headers=header)
    response.raise_for_status()
    response_json = response.json()

    return RepoModel(**response_json["model"])


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
    refresh_token: bool = False,
) -> RepoModel:
    """
    Downloads all files from specified model in the model repo.

    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: The architecture of the models e.g. mobilenet
    :param sub_architecture: The sub architecture of the model e.g. 1.0
    :param dataset: The dataset the model was trained on e.g. imagenet
    :param framework: The framework the model was trained on e.g. pytorch
    :param optimization_name: The level of optimization of the model e.g. base
    :param release_version: Optional param specifying the maximum supported release version for the models
    :param refresh_token: Forces a refresh of the authentication token
    :param overwrite: True to overwrite the file if it exists, False otherwise
    :param save_dir: The directory to save the model files to
        instead of the default cache dir
    :param save_path: The exact path to save the model files to instead of
        the default cache dir or save_dir
    :return: the downloaded model
    """
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

    model = RepoModel(**response_json["model"])
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
) -> RepoFile:
    """
    Downloads a file from specified model in the model repo.

    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: The architecture of the models e.g. mobilenet
    :param sub_architecture: The sub architecture of the model e.g. 1.0
    :param dataset: The dataset the model was trained on e.g. imagenet
    :param framework: The framework the model was trained on e.g. pytorch
    :param optimization_name: The level of optimization of the model e.g. base
    :param file_name: The name of the file being downloaded e.g. model.onnx
    :param release_version: Optional param specifying the maximum supported release version for the models
    :param refresh_token: Forces a refresh of the authentication token
    :param overwrite: True to overwrite the file if it exists, False otherwise
    :param save_dir: The directory to save the model files to
        instead of the default cache dir
    :param save_path: The exact path to save the model file to instead of
        the default cache dir or save_dir
    :return: a RepoFile for the downloaded model file
    """
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

    file_obj = RepoFile(**response_json["file"])

    save_location = file_obj.download(
        overwrite=overwrite, save_dir=save_dir, save_path=save_path
    )
    _LOGGER.info(f"Successfully saved file at {save_location}")
    return file_obj
