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
import yaml

from sparsezoo import Model
from sparsezoo.analyze import ModelAnalysis


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


def yaml_file():
    return str(Path(__file__).parent / "quantized_resnet.yaml")


def yaml_raw_str():
    return yaml.safe_load(yaml_file())


@pytest.mark.parametrize(
    "file_path, should_error",
    [
        (onnx_stub(), False),
        (onnx_deployment_dir(), False),
        (onnx_local_path(), False),
        (onnx_model(), False),
        # (yaml_file(), False),  # TODO: Failing, need to check creation from yaml
        # (yaml_raw_str(), False),  # TODO: Failing, need to check re-creation from yaml
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
