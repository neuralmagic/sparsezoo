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

from pathlib import Path

import onnx
import pytest

from sparsezoo import Model
from sparsezoo.analyze_v1 import ModelAnalysis


def onnx_stub():
    return (
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
        "imagenet/pruned95_quant-none"
    )


def onnx_deployment_dir():
    return Model(onnx_stub()).deployment.path


def onnx_local_path():
    return str(Path(onnx_deployment_dir()) / "model.onnx")


def onnx_model():
    return onnx.load(onnx_local_path())


@pytest.mark.parametrize(
    "file_path, should_error",
    [
        (onnx_stub(), False),
        (onnx_deployment_dir(), False),
        (onnx_local_path(), False),
        (onnx_model(), False),
        (1, True),
    ],
)
def test_create(file_path, should_error):
    if should_error:
        with pytest.raises(ValueError, match="Invalid"):
            ModelAnalysis.create(file_path)
    else:
        analysis = ModelAnalysis.create(file_path)
        assert isinstance(analysis, ModelAnalysis)


@pytest.mark.parametrize(
    "model_path",
    [
        onnx_local_path(),
    ],
)
def test_yaml_serialization(model_path, tmp_path):
    analysis = ModelAnalysis.create(file_path=model_path)

    yaml_file = str(tmp_path / "quantized-resnet.yaml")
    analysis.yaml(file_path=yaml_file)
    analysis.model_name = ""

    analysis_from_yaml = ModelAnalysis.create(file_path=yaml_file)
    analysis_from_yaml.model_name = ""

    expected = analysis.yaml()
    actual = analysis_from_yaml.yaml()
    assert actual == expected
