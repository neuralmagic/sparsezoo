"""
Code related to wrapping around API calls under api.neuralmagic.com/models/download
"""

import logging
from typing import List, Tuple

from sparsezoo.models import File, Model


__all__ = ["download_model_file", "download_model", "download_model_recipes"]


_LOGGER = logging.getLogger(__name__)


def download_model(
    domain: str,
    sub_domain: str,
    architecture: str,
    sub_architecture: str,
    dataset: str,
    framework: str,
    repo_source: str,
    optimization_name: str,
    release_version: str = None,
    overwrite: bool = False,
    save_dir: str = None,
    save_path: str = None,
    force_token_refresh: bool = False,
) -> Tuple[Model, str]:
    """
    Downloads all files from specified model in the model repo.

    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: The architecture of the models e.g. mobilenet
    :param sub_architecture: The sub architecture of the model e.g. 1.0
    :param dataset: The dataset the model was trained on e.g. imagenet
    :param framework: The framework the model was trained on e.g. pytorch
    :param repo_source: the source repo for the model e.g. neuralmagic, torchvision
    :param optimization_name: The level of optimization of the model e.g. base
    :param release_version: Optional param specifying the maximum supported
        release version for the models
    :param force_token_refresh: Forces a refresh of the authentication token
    :param overwrite: True to overwrite the file if it exists, False otherwise
    :param save_dir: The directory to save the model files to
        instead of the default cache dir
    :param save_path: The exact path to save the model files to instead of
        the default cache dir or save_dir
    :return: a tuple containing the Model and the save folder containing the model files
    """
    model = Model.get_downloadable_model(
        domain,
        sub_domain,
        architecture,
        sub_architecture,
        dataset,
        framework,
        repo_source,
        optimization_name,
        release_version=release_version,
        force_token_refresh=force_token_refresh,
    )
    save_location = model.download(
        overwrite=overwrite, save_dir=save_dir, save_path=save_path
    )
    _LOGGER.info(f"Successfully saved model files at {save_location}")
    return [model, save_location]


def download_model_file(
    domain: str,
    sub_domain: str,
    architecture: str,
    sub_architecture: str,
    dataset: str,
    framework: str,
    repo_source: str,
    optimization_name: str,
    file_name: str,
    release_version: str = None,
    overwrite: bool = False,
    save_dir: str = None,
    save_path: str = None,
    force_token_refresh: bool = False,
) -> Tuple[File, str]:
    """
    Downloads a file from specified model in the model repo.

    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: The architecture of the models e.g. mobilenet
    :param sub_architecture: The sub architecture of the model e.g. 1.0
    :param dataset: The dataset the model was trained on e.g. imagenet
    :param framework: The framework the model was trained on e.g. pytorch
    :param repo_source: the source repo for the model e.g. neuralmagic, torchvision
    :param optimization_name: The level of optimization of the model e.g. base
    :param file_name: The name of the file being downloaded e.g. model.onnx
    :param release_version: Optional param specifying the maximum supported
        release version for the models
    :param force_token_refresh: Forces a refresh of the authentication token
    :param overwrite: True to overwrite the file if it exists, False otherwise
    :param save_dir: The directory to save the model files to
        instead of the default cache dir
    :param save_path: The exact path to save the model file to instead of
        the default cache dir or save_dir
    :return: a tuple containing the File and the save location for the file
    """
    file_obj = File.get_downloadable_file(
        domain,
        sub_domain,
        architecture,
        sub_architecture,
        dataset,
        framework,
        repo_source,
        optimization_name,
        file_name,
        release_version=release_version,
        force_token_refresh=force_token_refresh,
    )
    save_location = file_obj.download(
        overwrite=overwrite, save_dir=save_dir, save_path=save_path
    )
    _LOGGER.info(f"Successfully saved file at {save_location}")
    return [file_obj, save_location]


def download_model_recipes(
    domain: str,
    sub_domain: str,
    architecture: str,
    sub_architecture: str,
    dataset: str,
    framework: str,
    repo_source: str,
    optimization_name: str,
    optimization_type: str = None,
    release_version: str = None,
    overwrite: bool = False,
    save_dir: str = None,
    save_path: str = None,
    force_token_refresh: bool = False,
) -> List[str]:
    """
    Downloads a file from specified model in the model repo.

    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: The architecture of the models e.g. mobilenet
    :param sub_architecture: The sub architecture of the model e.g. 1.0
    :param dataset: The dataset the model was trained on e.g. imagenet
    :param framework: The framework the model was trained on e.g. pytorch
    :param repo_source: the source repo for the model e.g. neuralmagic, torchvision
    :param optimization_name: The level of optimization of the model e.g. base
    :param release_version: Optional param specifying the maximum supported
        release version for the models
    :param force_token_refresh: Forces a refresh of the authentication token
    :param overwrite: True to overwrite the file if it exists, False otherwise
    :param save_dir: The directory to save the model files to
        instead of the default cache dir
    :param save_path: The exact path to save the model file to instead of
        the default cache dir or save_dir
    :return: a File for the downloaded model file
    """
    model = Model.get_downloadable_model(
        domain,
        sub_domain,
        architecture,
        sub_architecture,
        dataset,
        framework,
        repo_source,
        optimization_name,
        release_version=release_version,
        force_token_refresh=force_token_refresh,
    )
    save_location = model.download_recipe_files(
        overwrite=overwrite,
        save_dir=save_dir,
        save_path=save_path,
        optimization_type=optimization_type,
    )
    _LOGGER.info("Successfully saved model recipe files")
    return save_location
