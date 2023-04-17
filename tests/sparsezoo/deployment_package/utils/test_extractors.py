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

from unittest.mock import patch

import pytest

from sparsezoo import Model
from sparsezoo.deployment_package.utils.extractors import _batch_size_one_latency


_VNNI_STUB = (
    "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet"
    "/channel20_pruned75_quant-none-vnni"
)
_CV_STUBS = [
    _VNNI_STUB,
    "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet"
    "/pruned95_quant-none",
]

_NLP_STUBS = [
    "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad"
    "/pruned95_obs_quant-none"
]

_STUBS = _CV_STUBS + _NLP_STUBS


@pytest.mark.dependency()
def test_sparsezoo_version():
    from sparsezoo import version

    _version = version.__version__
    assert _version >= "1.2", f"Need atleast `sparsezoo>1.2` found {_version}"


@pytest.fixture()
def vnni():
    yield Model(_VNNI_STUB)


@pytest.fixture()
def model():
    stub = (
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet"
        "/pruned95_quant-none"
    )
    yield Model(source=stub)


@patch("sparsezoo.deployment_package_module.utils.extractors._throughput")
def test_latency_extractor(throughput_patch, model):
    throughput_patch.return_value = 0
    with pytest.raises(ValueError):
        _batch_size_one_latency(model=model)

    throughput_patch.return_value = 1
    actual = _batch_size_one_latency(model=model)
    assert actual == 1000 / throughput_patch.return_value
