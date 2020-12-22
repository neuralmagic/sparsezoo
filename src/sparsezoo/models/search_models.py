import os
import logging
from typing import Dict, List, Union

import requests

from sparsezoo.schemas import ModelSchema
from sparsezoo.utils import get_auth_header, BASE_API_URL


__all__ = ["search_model_repo"]


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


def search_model_repo(
    domain: str,
    sub_domain: str,
    search_args: Union[Dict, ModelRepoSearchArgs] = ModelRepoSearchArgs(),
    refresh_token: bool = False,
) -> List[ModelSchema]:
    header = get_auth_header(refresh_token=refresh_token)

    url = os.path.join(BASE_API_URL, "search", domain, sub_domain)

    if type(search_args) == dict:
        search_args = ModelRepoSearchArgs(**search_args)

    url = os.path.join(url, "?" + str(search_args))
    _LOGGER.info(f"Searching models from {url}")

    response_json = requests.get(url=url, headers=header).json()
    if "models" in response_json:
        models = [ModelSchema(**model) for model in response_json["models"]]
        return models
    else:
        raise Exception("Key 'models' not found in response")
