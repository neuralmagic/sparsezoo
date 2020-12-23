"""
Code related to wrapping around API calls under api.neuralmagic.com/models/search
"""

import os
import logging
from typing import List

import requests

from sparsezoo.schemas import RepoModel
from sparsezoo.utils import get_auth_header, BASE_API_URL


__all__ = ["search_models"]


_LOGGER = logging.getLogger(__name__)


class ModelRepoSearchArgs:
    def __init__(self, **kwargs):
        self.architecture = kwargs["architecture"] if "architecture" in kwargs else None
        self.sub_architecture = (
            kwargs["sub_architecture"] if "sub_architecture" in kwargs else None
        )
        self.dataset = kwargs["dataset"] if "dataset" in kwargs else None
        self.framework = kwargs["framework"] if "framework" in kwargs else None
        self.optimization_name = (
            kwargs["optimization_name"] if "optimization_name" in kwargs else None
        )
        self.release_version = (
            kwargs["release_version"] if "release_version" in kwargs else None
        )
        self.page = kwargs["page"] if "page" in kwargs else 1
        self.page_length = kwargs["page_length"] if "page_length" in kwargs else 20

    def __str__(self) -> str:
        args = []
        for key, value in vars(self).items():
            if type(value) == list:
                args += [f"{key}={item}" for item in value]
            elif value is not None:
                args.append(f"{key}={value}")
            else:
                continue
        return "&".join(args)


def search_models(
    domain: str,
    sub_domain: str,
    architecture: str = None,
    sub_architecture: str = None,
    dataset: str = None,
    framework: str = None,
    optimization_name: str = None,
    release_version: str = None,
    page: int = 1,
    page_length: int = 20,
    force_token_refresh: bool = False,
) -> List[RepoModel]:
    """
    Search model repo for any models matching the search criteria.

    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: Optional param specifying the architecture of the models e.g. mobilenet
    :param sub_architecture: Optional param specifying the sub architecture of the models e.g. 1.0
    :param dataset: Optional param specifying the dataset the models were trained on e.g. imagenet
    :param framework: Optional param specifying the framework the models were trained on e.g. pytorch
    :param optimization_name: Optional param specifying the level of optimization of the models e.g. base
    :param release_version: Optional param specifying the maximum supported release version for the models
    :param force_token_refresh: Forces a refresh of the authentication token
    :return: list of models matching provided criterias
    """
    header = get_auth_header(force_token_refresh=force_token_refresh)

    url = os.path.join(BASE_API_URL, "search", domain, sub_domain)

    search_args = ModelRepoSearchArgs(
        architecture=architecture,
        sub_architecture=sub_architecture,
        dataset=dataset,
        framework=framework,
        optimization_name=optimization_name,
        release_version=release_version,
        page=page,
        page_length=page_length,
    )

    url = os.path.join(url, "?" + str(search_args))
    _LOGGER.info(f"Searching models from {url}")

    response_json = requests.get(url=url, headers=header).json()
    models = [RepoModel(**model) for model in response_json["models"]]
    return models
