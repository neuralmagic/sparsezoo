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
Code related to base functionality for making requests
"""

from typing import Any, List, Union


__all__ = ["BASE_API_URL", "ModelArgs"]

BASE_API_URL = "https://api.neuralmagic.com/models"


class ModelArgs:
    """
    Arguments for making requests into the sparsezoo

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
    :param repo: the source repo for the model the object belongs to;
        e.g. sparseml, torchvision
    :param dataset: The dataset the model the object belongs to was trained on;
        e.g. imagenet, cifar10
    :param training_scheme: The training scheme used on the model the object belongs
        to if any; e.g. augmented
    :param optim_name: The name describing the optimization of the model
        the object belongs to, e.g. base, pruned, pruned_quant,
    :param optim_category: The degree of optimization of the model the object
        belongs to; e.g. none, conservative (~100% baseline metric),
        moderate (>=99% baseline metric), aggressive (<99% baseline metric)
    :param optim_target: The deployment target of optimization of the model
        the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
    :param release_version: The sparsezoo release version for the model
    """

    def __init__(
        self,
        domain: Union[str, None] = None,
        sub_domain: Union[str, None] = None,
        architecture: Union[str, List[str], None] = None,
        sub_architecture: Union[str, List[str], None] = None,
        framework: Union[str, List[str], None] = None,
        repo: Union[str, List[str], None] = None,
        dataset: Union[str, List[str], None] = None,
        training_scheme: Union[str, List[str], None] = None,
        optim_name: Union[str, List[str], None] = None,
        optim_category: Union[str, List[str], None] = None,
        optim_target: Union[str, List[str], None] = None,
        release_version: Union[str, Any, None] = None,
        **kwargs,
    ):
        self._domain = domain
        self._sub_domain = sub_domain
        self._architecture = architecture
        self._sub_architecture = sub_architecture
        self._framework = framework
        self._repo = repo
        self._dataset = dataset
        self._training_scheme = training_scheme
        self._optim_name = optim_name
        self._optim_category = optim_category
        self._optim_target = optim_target
        self._release_version = release_version

    @property
    def domain(self) -> Union[str, None]:
        """
        :return: The domain of the model the object belongs to;
            e.g. cv, nlp
        """
        return self._domain

    @property
    def sub_domain(self) -> Union[str, None]:
        """
        :return: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        """
        return self._sub_domain

    @property
    def architecture(self) -> Union[str, List[str], None]:
        """
        :return: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        """
        return self._architecture

    @property
    def sub_architecture(self) -> Union[str, List[str], None]:
        """
        :return: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        """
        return self._sub_architecture

    @property
    def framework(self) -> Union[str, List[str], None]:
        """
        :return: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        """
        return self._framework

    @property
    def repo(self) -> Union[str, List[str], None]:
        """
        :return: the source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        """
        return self._repo

    @property
    def dataset(self) -> Union[str, List[str], None]:
        """
        :return: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        """
        return self._dataset

    @property
    def training_scheme(self) -> Union[str, List[str], None]:
        """
        :return: The training scheme used on the model the object belongs to if any;
            e.g. augmented
        """
        return self._training_scheme

    @property
    def optim_name(self) -> Union[str, List[str], None]:
        """
        :return: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        """
        return self._optim_name

    @property
    def optim_category(self) -> Union[str, List[str], None]:
        """
        :return: The degree of optimization of the model the object belongs to;
            e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        """
        return self._optim_category

    @property
    def optim_target(self) -> Union[str, List[str], None]:
        """
        :return: The deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        """
        return self._optim_target

    @property
    def release_version(self) -> Union[str, None]:
        """
        :return: The sparsezoo release version for the model
        """
        return self._release_version

    @property
    def architecture_id(self) -> str:
        """
        :return: Unique id for the model architecture containing both the
            architecture and sub_architecture
        """
        if not self.architecture:
            return ""

        if not self.sub_architecture:
            return f"{self.architecture}"

        return f"{self.architecture}-{self.sub_architecture}"

    @property
    def training_id(self) -> str:
        """
        :return: Unique id for how the model was trained containing both the
            dataset and training_scheme
        """
        if not self.dataset:
            return ""

        if not self.training_scheme:
            return f"{self.dataset}"

        return f"{self.dataset}-{self.training_scheme}"

    @property
    def optimization_id(self) -> str:
        """
        :return: Unique id for how the model was optimized containing the
            optim_name, optim_category, optim_target
        """
        if not self.optim_name:
            return ""

        if not self.optim_category:
            return f"{self.optim_name}"

        if not self.optim_target:
            return f"{self.optim_name}-{self.optim_category}"

        return f"{self.optim_name}-{self.optim_category}-{self.optim_target}"

    @property
    def model_url_root(self) -> str:
        """
        :return: root path for where the model is located in the sparsezoo
        """
        if not self.domain:
            return ""

        if not self.sub_domain:
            return f"{self.domain}"

        return f"{self.domain}/{self.sub_domain}"

    @property
    def stub(self) -> str:
        """
        :return: full path for where the model is located in the sparsezoo
        """
        return "/".join(
            [
                self.model_url_root,
                self.architecture_id,
                f"{self.framework}" if self.framework else "",
                f"{self.repo}" if self.repo else "",
                self.training_id,
                self.optimization_id,
            ]
        )

    @property
    def model_url_args(self) -> List[str]:
        """
        :return: arguments for searching in the sparsezoo
        """
        args = []

        for key in [
            "architecture",
            "sub_architecture",
            "framework",
            "repo",
            "dataset",
            "training_scheme",
            "optim_name",
            "optim_category",
            "optim_target",
        ]:
            value = getattr(self, key)

            if value and isinstance(value, List):
                args.extend([f"{key}={item}" for item in value])
            elif value:
                args.append(f"{key}={value}")

        return args
