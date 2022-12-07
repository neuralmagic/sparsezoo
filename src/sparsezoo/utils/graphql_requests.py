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

import logging
from typing import Dict, Union

import requests

from sparsezoo import Model
from sparsezoo.search import model_args_to_stub
from sparsezoo.utils import MODELS_API_URL


__all__ = ["download_get_request", "search_model_get_request"]

_LOGGER = logging.getLogger(__name__)


def download_get_request(
    args: str,
    base_url: str = MODELS_API_URL,
    sub_path: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Get a downloadable object from the sparsezoo for any objects matching the args

    The path called has structure:
        [base_url]/download/[args.stub]/{sub_path}

    :param base_url: the base url
    :param args: the model args describing what should be downloaded for
    :param sub_path: the sub path from the model path if any e.g.
        file_name for models api or recipe_type for the recipes api
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    pass


def search_model_get_request(
    args: Dict[str, str],
    page: int = 1,
    page_length: int = 20,
    force_token_refresh: bool = False,
) -> Dict:
    """
    Search the sparsezoo for any models matching the args
    :param args: the dictionary describing what should be searched for
    :param page: the page of values to get
    :param page_length: the page length of values to get
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: the json response as a dict
    """
    # url = "https://staging-api.neuralmagic.com/v2/graphql"
    url = "https://api.neuralmagic.com/v2/graphql"
    # url = "http://0.0.0.0:8000/prod/graphql"
    body = """
    {{
        models {request_args}
            {{
                architecture
                base
                baseModel
                displayDescription
                displayName
                domain
                framework
                hidden
                modelId
                repo
                repoName
                repoNamespace
                sourceDataset
                sparseTag
                subArchitecture
                task
                taskDataset
                trainingDataset
                trainingScheme
            }}
    }}
    """
    request_args = ""
    for key, value in args.items():
        if value is not None:
            request_args += f'{key}: "{value}",'
    request_args = "(" + request_args + ")"

    response_json = requests.post(
        url=url, json={"query": body.format(request_args=request_args)}
    ).json()

    for model_args in response_json["data"]["models"]:
        print(model_args_to_stub(**model_args))

    models = [
        Model(model_args_to_stub(**model_args))
        for model_args in response_json["data"]["models"]
    ]
    print(models)

    # models = [Model(model_args_to_stub(**model_args)) for model_args in response_json["data"]["models"]]
    # print(models)

    pass
