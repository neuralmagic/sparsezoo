"""
Code related to wrapping around API calls under api.neuralmagic.com/models/search
"""

import logging
import os
from typing import List

import requests
from sparsezoo.models import Model
from sparsezoo.utils import BASE_API_URL, get_auth_header


__all__ = ["search_models", "ModelRepoSearchArgs"]


_LOGGER = logging.getLogger(__name__)


class ModelRepoSearchArgs:
    """
    The optional search arguments for a GET request for
    https://api.neuralmagic.com/models/search

    :param architecture: The architecture of the models e.g. mobilenet
    :param sub_architecture: The sub architecture of the models e.g. 1.0
    :param dataset: The dataset the models were trained on e.g. imagenet
    :param framework: The framework the models were trained with e.g. pytorch
    :param optimization_name: The level of optimization of the models e.g. base
    :param release_version: The model repo release version models were released with
    :param repo_source: the source repo for the model
    :param page: The page to request results for
    :param page_length: The number of results to show
    """

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
        self.repo_source = kwargs["repo_source"] if "repo_source" in kwargs else None
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
    repo_source: str = None,
    page: int = 1,
    page_length: int = 20,
    force_token_refresh: bool = False,
) -> List[Model]:
    """
    Search model repo for any models matching the search criteria.

    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: Optional param specifying the architecture of
        the models e.g. mobilenet
    :param sub_architecture: Optional param specifying the sub architecture
        of the models e.g. 1.0
    :param dataset: Optional param specifying the dataset the models were
        trained on e.g. imagenet
    :param framework: Optional param specifying the framework the models
        were trained on e.g. pytorch
    :param repo_source: the source repo for the model
    :param optimization_name: Optional param specifying the level of
        optimization of the models e.g. base
    :param release_version: Optional param specifying the maximum supported
        release version for the models
    :param force_token_refresh: Forces a refresh of the authentication token
    :return: list of models matching provided criterias
    """
    if not page > 0:
        raise Exception("'page' value must be > 0")
    if not page_length > 0:
        raise Exception("'page_length' value must be > 0")
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
        repo_source=repo_source,
    )

    url = os.path.join(url, "?" + str(search_args))
    _LOGGER.info(f"Searching models from {url}")

    response_json = requests.get(url=url, headers=header).json()
    models = [Model(**model) for model in response_json["models"]]
    return models
