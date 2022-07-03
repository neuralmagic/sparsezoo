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
Helper fixtures and functions for testing sparsezoo.analysis
"""

import onnx
import pytest

from sparsezoo import Zoo
from sparsezoo.analysis import ModelAnalysis


__all__ = [
    "model_paths",
    "get_test_model_names",
    "get_expected_analysis",
    "get_generated_analysis",
    "get_model_onnx",
    "get_model_and_node",
]

_MODEL_PATHS = {
    "yolact_none": {
        "stub": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/" "base-none",
        "truth": "tests/sparsezoo/analysis/yolact_none.yaml",
    },
    "mobilenet_v1_pruned_moderate": {
        "stub": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
        "imagenet/pruned-moderate",
        "truth": "tests/sparsezoo/analysis/mobilenet_v1_pruned_moderate.yaml",
    },
    "bert_pruned_quantized": {
        "stub": "zoo:nlp/question_answering/bert-base/pytorch/huggingface/"
        "squad/12layer_pruned80_quant-none-vnni",
        "truth": "tests/sparsezoo/analysis/bert_pruned_quantized.yaml",
    },
    "resnet50_pruned_quantized": {
        "stub": "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
        "imagenet/pruned85_quant-none-vnni",
        "truth": "tests/sparsezoo/analysis/resnet50_pruned_quantized.yaml",
    },
}


@pytest.fixture(scope="session")
def model_paths():
    return _MODEL_PATHS


def get_test_model_names():
    return _MODEL_PATHS.keys()


@pytest.fixture(scope="session")
def get_expected_analysis():
    model_truth_analyses = {}
    for model_name in _MODEL_PATHS.keys():
        model_truth_path = _MODEL_PATHS[model_name]["truth"]
        analysis = ModelAnalysis.parse_yaml_file(model_truth_path)
        model_truth_analyses[model_name] = analysis

    def _get_expected_analysis(model_name):
        return model_truth_analyses[model_name]

    return _get_expected_analysis


@pytest.fixture(scope="session")
def get_generated_analysis():
    model_generated_analyses = {}
    for model_name in _MODEL_PATHS.keys():
        model_stub = _MODEL_PATHS[model_name]["stub"]
        model = Zoo.load_model_from_stub(model_stub)
        model.onnx_file.download()
        onnx_path = model.onnx_file.downloaded_path()
        analysis = ModelAnalysis.from_onnx(onnx_path)
        model_generated_analyses[model_name] = analysis

    def _get_generated_analysis(model_name):
        return model_generated_analyses[model_name]

    return _get_generated_analysis


@pytest.fixture(scope="session")
def get_model_onnx():
    model_onnxs = {}
    for model_name in _MODEL_PATHS.keys():
        model_stub = _MODEL_PATHS[model_name]["stub"]
        model = Zoo.load_model_from_stub(model_stub)
        model.onnx_file.download()
        onnx_path = model.onnx_file.downloaded_path()
        model_onnx = onnx.load(onnx_path)
        model_onnxs[model_name] = model_onnx

    def _get_model_onnx(model_name):
        return model_onnxs[model_name]

    return _get_model_onnx


@pytest.fixture()
def get_model_and_node(get_model_onnx):
    def _get_model_and_node(model_name, node_name):
        model = get_model_onnx(model_name)
        return (
            model,
            [node for node in list(model.graph.node) if node.name == node_name][0],
        )

    return _get_model_and_node
