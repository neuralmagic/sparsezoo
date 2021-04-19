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

"""
MobileNet models:
    https://arxiv.org/abs/1704.04861
    https://arxiv.org/abs/1801.04381
"""

from typing import Union

from sparsezoo.models.zoo import Zoo
from sparsezoo.objects import Model


__all__ = ["mobilenet_v1", "mobilenet_v2"]


def mobilenet_v1(
    framework: str = "pytorch",
    repo: str = "sparseml",
    dataset: str = "imagenet",
    training_scheme: Union[str, None] = None,
    sparse_name: str = "base",
    sparse_category: str = "none",
    sparse_target: Union[str, None] = None,
    override_folder_name: Union[str, None] = None,
    override_parent_path: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Model:
    """
    Convenience function for getting a mobilenet_v1 model

    :param framework: The framework the model the object belongs to was trained on;
        e.g. pytorch, tensorflow
    :param repo: the source repo for the model the object belongs to;
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
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: The created model
    """
    return Zoo.load_model(
        domain="cv",
        sub_domain="classification",
        architecture="mobilenet_v1",
        sub_architecture="1.0",
        framework=framework,
        repo=repo,
        dataset=dataset,
        training_scheme=training_scheme,
        sparse_name=sparse_name,
        sparse_category=sparse_category,
        sparse_target=sparse_target,
        override_folder_name=override_folder_name,
        override_parent_path=override_parent_path,
        force_token_refresh=force_token_refresh,
    )


def mobilenet_v2(
    framework: str = "pytorch",
    repo: str = "sparseml",
    dataset: str = "imagenet",
    training_scheme: Union[str, None] = None,
    sparse_name: str = "base",
    sparse_category: str = "none",
    sparse_target: Union[str, None] = None,
    override_folder_name: Union[str, None] = None,
    override_parent_path: Union[str, None] = None,
    force_token_refresh: bool = False,
) -> Model:
    """
    Convenience function for getting a mobilenet_v2 model

    :param framework: The framework the model the object belongs to was trained on;
        e.g. pytorch, tensorflow
    :param repo: the source repo for the model the object belongs to;
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
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: The created model
    """
    return Zoo.load_model(
        domain="cv",
        sub_domain="classification",
        architecture="mobilenet_v2",
        sub_architecture="1.0",
        framework=framework,
        repo=repo,
        dataset=dataset,
        training_scheme=training_scheme,
        sparse_name=sparse_name,
        sparse_category=sparse_category,
        sparse_target=sparse_target,
        override_folder_name=override_folder_name,
        override_parent_path=override_parent_path,
        force_token_refresh=force_token_refresh,
    )
