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
Helper fixtures and functions for testing sparsezoo.analyze
"""

import os

import onnx
import pytest

from sparsezoo import Model
from sparsezoo.analyze_v1 import ModelAnalysis
from sparsezoo.utils.graph_editor import ONNXGraph


__all__ = [
    "model_paths",
    "get_test_model_names",
    "get_expected_analysis",
    "get_generated_analysis",
    "get_model_graph",
    "get_model_graph_and_node",
]

_MODEL_PATHS = {
    "yolact_none": {
        "stub": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none",
        "truth": f"{os.path.dirname(__file__)}/yolact_none.json",
    },
    "mobilenet_v1_pruned_moderate": {
        "stub": (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
            "imagenet/pruned-moderate"
        ),
        "truth": f"{os.path.dirname(__file__)}/mobilenet_v1_pruned_moderate.json",
    },
    "bert_pruned_quantized": {
        "stub": (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/"
            "pruned80_quant-none-vnni"
        ),
        "truth": f"{os.path.dirname(__file__)}/bert_pruned80_quant-none-vnni.json",
    },
    "resnet50_pruned_quantized": {
        "stub": (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned85_quant-none-vnni"
        ),
        "truth": f"{os.path.dirname(__file__)}/resnet50_pruned_quantized.json",
    },
}


@pytest.fixture(scope="session")
def model_paths():
    return _MODEL_PATHS


def get_test_model_names():
    return _MODEL_PATHS.keys()


@pytest.fixture(scope="session")
def get_generated_analysis():
    model_generated_analyses = {}
    for model_name in _MODEL_PATHS.keys():
        model_stub = _MODEL_PATHS[model_name]["stub"]
        model = Model(model_stub)

        onnx_path = model.onnx_model.path
        analysis = ModelAnalysis.from_onnx(onnx_path)
        model_generated_analyses[model_name] = analysis

    def _get_generated_analysis(model_name):
        return model_generated_analyses[model_name]

    return _get_generated_analysis


@pytest.fixture(scope="session")
def get_expected_analysis(get_generated_analysis):
    model_truth_analyses = {}
    for model_name in _MODEL_PATHS.keys():
        model_truth_path = _MODEL_PATHS[model_name]["truth"]

        # if env variable set, automatically update truth files
        if os.getenv("NM_GENERATE_ANALYSIS_TEST_FILES", False):
            model_analysis = get_generated_analysis(model_name)
            with open(model_truth_path, "w") as truth_file:
                truth_file.write(model_analysis.json())

        # read truth file
        with open(model_truth_path, "r") as truth_file:
            analysis = ModelAnalysis.parse_raw(truth_file.read())
        model_truth_analyses[model_name] = analysis

    def _get_expected_analysis(model_name):
        return model_truth_analyses[model_name]

    return _get_expected_analysis


@pytest.fixture(scope="session")
def get_model_graph():
    model_graphs = {}
    for model_name in _MODEL_PATHS.keys():
        model_stub = _MODEL_PATHS[model_name]["stub"]
        model = Model(model_stub)
        onnx_path = model.onnx_model.path
        model_onnx = onnx.load(onnx_path)
        model_graph = ONNXGraph(model_onnx)
        model_graphs[model_name] = model_graph

    def _get_model_graph(model_name):
        return model_graphs[model_name]

    return _get_model_graph


@pytest.fixture(scope="session")
def get_model_graph_and_node(get_model_graph):
    def _get_model_graph_and_node(model_name, node_name):
        model_graph = get_model_graph(model_name)
        return (
            model_graph,
            [node for node in list(model_graph.nodes) if node.name == node_name][0],
        )

    return _get_model_graph_and_node
