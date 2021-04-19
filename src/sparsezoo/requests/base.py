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

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from sparsezoo.utils import convert_to_bool


__all__ = [
    "BASE_API_URL",
    "ModelArgs",
    "MODELS_API_URL",
    "SPARSEZOO_TEST_MODE",
    "RecipeArgs",
    "RECIPES_API_URL",
    "parse_zoo_stub",
]


SPARSEZOO_TEST_MODE = convert_to_bool(os.getenv("SPARSEZOO_TEST_MODE"))

BASE_API_URL = (
    os.getenv("SPARSEZOO_API_URL")
    if os.getenv("SPARSEZOO_API_URL")
    else "https://api.neuralmagic.com"
)
MODELS_API_URL = f"{BASE_API_URL}/models"
RECIPES_API_URL = f"{BASE_API_URL}/recipes"

# optional prefix for stubs
ZOO_STUB_PREFIX = "zoo:"


def parse_zoo_stub(
    stub: str, valid_params: Optional[List[str]] = None
) -> Tuple[str, Dict[str, str]]:
    """
    :param stub: A SparseZoo model stub. i.e. 'model/stub/path',
        'zoo:model/stub/path', 'zoo:model/stub/path?param1=value1&param2=value2'
    :param valid_params: list of expected parameter names to be encoded in the
        stub. Will raise a warning if any unexpected param names are given. Leave
        as None to not raise any warnings. Default is None
    :return: the parsed base stub and a dictionary of parameter names and their values
    """
    # strip optional zoo stub prefix
    if stub.startswith(ZOO_STUB_PREFIX):
        stub = stub[len(ZOO_STUB_PREFIX) :]

    if "?" not in stub:
        return stub, {}

    stub_parts = stub.split("?")
    if len(stub_parts) > 2:
        raise ValueError(
            "Invalid SparseZoo stub, query string must be preceded by only one '?'"
            f"given {stub}"
        )
    stub, params = stub_parts
    params = dict(param.split("=") for param in params.split("&"))

    if valid_params is not None and any(param not in valid_params for param in params):
        warnings.warn(
            f"Invalid query string for stub {stub} valid params include {valid_params},"
            f" given {list(params.keys())}"
        )

    return stub, params


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
    :param sparse_name: The name describing the sparsification of the model
        the object belongs to, e.g. base, pruned, pruned_quant,
    :param sparse_category: The degree of sparsification of the model the object
        belongs to; e.g. none, conservative (~100% baseline metric),
        moderate (>=99% baseline metric), aggressive (<99% baseline metric)
    :param sparse_target: The deployment target of sparsification of the model
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
        sparse_name: Union[str, List[str], None] = None,
        sparse_category: Union[str, List[str], None] = None,
        sparse_target: Union[str, List[str], None] = None,
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
        self._sparse_name = sparse_name
        self._sparse_category = sparse_category
        self._sparse_target = sparse_target
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
    def sparse_name(self) -> Union[str, List[str], None]:
        """
        :return: The name describing the sparsification of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        """
        return self._sparse_name

    @property
    def sparse_category(self) -> Union[str, List[str], None]:
        """
        :return: The degree of sparsification of the model the object belongs to;
            e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        """
        return self._sparse_category

    @property
    def sparse_target(self) -> Union[str, List[str], None]:
        """
        :return: The deployment target of sparsification of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        """
        return self._sparse_target

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
    def sparse_id(self) -> str:
        """
        :return: Unique id for how the model was sparse containing the
            sparse_name, sparse_category, sparse_target
        """
        if not self.sparse_name:
            return ""

        if not self.sparse_category:
            return f"{self.sparse_name}"

        if not self.sparse_target:
            return f"{self.sparse_name}-{self.sparse_category}"

        return f"{self.sparse_name}-{self.sparse_category}-{self.sparse_target}"

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
                self.sparse_id,
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
            "sparse_name",
            "sparse_category",
            "sparse_target",
        ]:
            value = getattr(self, key)

            if value and isinstance(value, List):
                args.extend([f"{key}={item}" for item in value])
            elif value:
                args.append(f"{key}={value}")

        return args


class RecipeArgs(ModelArgs):
    """
    Arguments for making recipe requests into the sparsezoo

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
    :param sparse_name: The name describing the sparsification of the model
        the object belongs to, e.g. base, pruned, pruned_quant,
    :param sparse_category: The degree of sparsification of the model the object
        belongs to; e.g. none, conservative (~100% baseline metric),
        moderate (>=99% baseline metric), aggressive (<99% baseline metric)
    :param sparse_target: The deployment target of sparsification of the model
        the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
    :param release_version: The sparsezoo release version for the model
    :param recipe_type: The recipe type; e.g. original, transfer_learn
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
        sparse_name: Union[str, List[str], None] = None,
        sparse_category: Union[str, List[str], None] = None,
        sparse_target: Union[str, List[str], None] = None,
        release_version: Union[str, Any, None] = None,
        recipe_type: Union[str, None] = None,
        **kwargs,
    ):
        super(RecipeArgs, self).__init__(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            sparse_name=sparse_name,
            sparse_category=sparse_category,
            sparse_target=sparse_target,
            release_version=release_version,
            **kwargs,
        )
        self._recipe_type = recipe_type

    @property
    def recipe_type(self) -> Union[str, None]:
        """
        :return: The recipe type; e.g. original, transfer_learn
        """
        return self._recipe_type

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
            "sparse_name",
            "sparse_category",
            "sparse_target",
            "recipe_type",
        ]:
            value = getattr(self, key)

            if value and isinstance(value, List):
                args.extend([f"{key}={item}" for item in value])
            elif value:
                args.append(f"{key}={value}")

        return args
