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
from typing import List, Union

from sparsezoo import Model
from sparsezoo.utils import search_model_get_request


__all__ = ["search_models", "model_args_to_stub"]


def search_models(
    domain: str,
    sub_domain: str,
    architecture: Union[str, None] = None,
    sub_architecture: Union[str, None] = None,
    framework: Union[str, None] = None,
    repo: Union[str, None] = None,
    dataset: Union[str, None] = None,
    training_scheme: Union[str, None] = None,
    sparse_name: Union[str, None] = None,
    sparse_category: Union[str, None] = None,
    sparse_target: Union[str, None] = None,
    release_version: Union[str, None] = None,
    page: int = 1,
    page_length: int = 20,
    force_token_refresh: bool = False,
    return_stubs: bool = False,
) -> List[Model]:
    """
    Obtains a list of Models matching the search parameters
    :param domain: The domain of the model the object belongs to;
        e.g. cv, nlp
    :param sub_domain: The sub domain of the model the object belongs to;
        e.g. classification, segmentation
    :param architecture: The architecture of the model the object belongs to;
        e.g. resnet_v1, mobilenet_v1
    :param sub_architecture: The sub architecture (scaling factor) of the model
        the object belongs to; e.g. 50, 101, 152
    :param framework: The framework the model the object belongs to was trained on;
        e.g. pytorch, tensorflow
    :param repo: The source repo for the model the object belongs to;
        e.g. sparseml, torchvision
    :param dataset: The dataset the model the object belongs to was trained on;
        e.g. imagenet, cifar10
    :param training_scheme: The training scheme used on the model the object
        belongs to if any; e.g. augmented
    :param sparse_name: The name describing the sparsification of the model
        the object belongs to, e.g. base, pruned, pruned_quant
    :param sparse_category: The degree of sparsification of the model the object
        belongs to; e.g. none, conservative (~100% baseline metric),
        moderate (>=99% baseline metric), aggressive (<99% baseline metric)
    :param sparse_target: The deployment target of sparsification of the model
        the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
    :param release_version: The sparsezoo release version for the model
    :param page: the page of values to get
    :param page_length: the page length of values to get
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :param return_stubs: if True, found models will be returned as stubs only
        instead of retrieving full info for each found model. Default False
    :return: The requested list of Model instances
    """
    args = {
        "domain": domain,
        "sub_domain": sub_domain,
        "architecture": architecture,
        "sub_architecture": sub_architecture,
        "framework": framework,
        "repo": repo,
        "dataset": dataset,
        "training_scheme": training_scheme,
        "sparse_name": sparse_name,
        "sparse_category": sparse_category,
        "sparse_target": sparse_target,
        "release_version": release_version,
    }

    args = {k: v for k, v in args.items() if v is not None}

    logging.debug(f"Search_models: searching models with args {args}")
    response_json = search_model_get_request(
        args=args,
        page=page,
        page_length=page_length,
        force_token_refresh=force_token_refresh,
    )

    stubs = [model_args_to_stub(**model_dict) for model_dict in response_json["models"]]

    if return_stubs:
        return stubs

    return [Model(stub) for stub in stubs]


def model_args_to_stub(**kwargs) -> str:

    domain = kwargs.get("domain")
    sub_domain = kwargs.get("sub_domain")
    architecture = kwargs.get("architecture")
    sub_architecture = kwargs.get("sub_architecture")
    framework = kwargs.get("framework")
    repo = kwargs.get("repo")
    dataset = kwargs.get("dataset")
    sparse_tag = kwargs.get("sparse_tag")

    sparse_name = kwargs.get("sparse_name")
    sparse_category = kwargs.get("sparse_category")

    if not sparse_tag:
        sparse_tag = sparse_name + "-" + sparse_category

    if sub_architecture is not None:
        sub_architecture = "-" + sub_architecture

    stub = (
        f"zoo:{domain if domain is not None else ''}/"
        f"{sub_domain if sub_domain is not None else ''}/"
        f"{architecture if architecture is not None else ''}"
        f"{sub_architecture if sub_architecture is not None else ''}/"
        f"{framework if framework is not None else ''}/"
        f"{repo if repo is not None else ''}/"
        f"{dataset if dataset is not None else ''}/"
        f"{sparse_tag if sparse_tag is not None else ''}"
    )

    return stub
