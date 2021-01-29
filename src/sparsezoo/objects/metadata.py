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
Code related to metadata for models as stored in the sparsezoo
"""

from typing import Union

from sparsezoo.objects.base import BaseObject
from sparsezoo.objects.release_version import ReleaseVersion
from sparsezoo.requests import ModelArgs


__all__ = ["ModelMetadata"]


class ModelMetadata(BaseObject, ModelArgs):
    """
    Metadata to describe a model as stored in the sparsezoo

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
    :param optim_name: The name describing the optimization of the model
        the object belongs to, e.g. base, pruned, pruned_quant
    :param optim_category: The degree of optimization of the model the object
        belongs to; e.g. none, conservative (~100% baseline metric),
        moderate (>=99% baseline metric), aggressive (<99% baseline metric)
    :param optim_target: The deployment target of optimization of the model
        the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
    :param release_version: The sparsezoo release version for the model
    :param model_id: The id for the model as stored in the cloud
    :param base_model: The id of the base model as stored in the cloud
    :param user_id: The id of the user who uploaded the model as stored in the cloud
    """

    def __init__(
        self,
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: Union[str, None],
        framework: str,
        repo: str,
        dataset: str,
        training_scheme: Union[str, None],
        optim_name: str,
        optim_category: str,
        optim_target: Union[str, None],
        release_version: ReleaseVersion,
        model_id: str,
        base_model: str,
        user_id: str,
        **kwargs,
    ):
        super(BaseObject, self).__init__(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
            **kwargs,
        )
        super(ModelMetadata, self).__init__(**kwargs)
        self._model_id = model_id
        self._base_model = base_model
        self._user_id = user_id

    @property
    def model_id(self) -> str:
        """
        :return: The id for the model as stored in the cloud
        """
        return self._model_id

    @property
    def base_model(self) -> str:
        """
        :return: The id of the base model as stored in the cloud
        """
        return self._base_model

    @property
    def user_id(self) -> str:
        """
        :return: The id of the user who uploaded the model as stored in the cloud
        """
        return self._user_id
