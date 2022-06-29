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

import pytest

from sparsezoo import Zoo
from sparsezoo.analysis import ModelAnalysis


_MODEL_NAMES = [
    "yolact_none",
    "mobilenet_v1_pruned_moderate",
    "bert_pruned_quantized",
    "resnet50_pruned_quantized",
]


@pytest.fixture()
def margin_of_error():
    return 0.05


@pytest.fixture(scope="session")
def model_paths():
    return {
        "yolact_none": {
            "stub": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/"
            "base-none",
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
def get_generated_analysis(model_paths):
    model_generated_analyses = {}
    for model_name in model_paths.keys():
        model_stub = model_paths[model_name]["stub"]
        model = Zoo.load_model_from_stub(model_stub)
        model.onnx_file.download()
        onnx_path = model.onnx_file.downloaded_path()
        analysis = ModelAnalysis.from_onnx_model(onnx_path)
        model_generated_analyses[model_name] = analysis

    def _get_generated_analysis(model_name):
        return model_generated_analyses[model_name]

    return _get_generated_analysis


@pytest.fixture(scope="session")
def get_expected_analysis(model_paths):
    model_truth_analyses = {}
    for model_name in model_paths.keys():
        model_truth_path = model_paths[model_name]["truth"]
        analysis = ModelAnalysis.parse_yaml_file(model_truth_path)
        model_truth_analyses[model_name] = analysis

    def _get_expected_analysis(model_name):
        return model_truth_analyses[model_name]

    return _get_expected_analysis


def pytest_generate_tests(metafunc):
    metafunc.parametrize("model_name", _MODEL_NAMES)


def test_layer_counts(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.layer_counts == expected_model_analysis.layer_counts


def test_non_parameterized_operator_counts(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.non_parameterized_operator_counts
        == expected_model_analysis.non_parameterized_operator_counts
    )


def test_num_dense_ops(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_dense_ops == expected_model_analysis.num_dense_ops


def test_num_sparse_ops(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_sparse_ops == expected_model_analysis.num_sparse_ops


def test_num_sparse_layers(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_sparse_layers == expected_model_analysis.num_sparse_layers


def test_num_quantized_layers(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.num_quantized_layers
        == expected_model_analysis.num_quantized_layers
    )


def test_num_parameters(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_parameters == expected_model_analysis.num_parameters


def test_num_sparse_parameters(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.num_sparse_parameters
        == expected_model_analysis.num_sparse_parameters
    )


def test_num_four_blocks(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_four_blocks == expected_model_analysis.num_four_blocks


def test_num_sparse_four_blocks(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.num_sparse_four_blocks
        == expected_model_analysis.num_sparse_four_blocks
    )


def test_average_sparsity(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.average_sparsity == pytest.approx(
        expected_model_analysis.average_sparsity, abs=margin_of_error
    )


def test_average_four_block_sparsity(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.average_four_block_sparsity == pytest.approx(
        expected_model_analysis.average_four_block_sparsity, abs=margin_of_error
    )


def test_node_analyses(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert len(model_analysis.layers) == len(expected_model_analysis.layers)
    for i in range(len(model_analysis.layers)):
        node_analysis = model_analysis.layers[i]
        expected_node_analysis = model_analysis.layers[i]
        assert node_analysis.name == expected_node_analysis.name


def test_model_analysis_yaml(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    model_yaml = model_analysis.yaml()
    model_from_yaml = ModelAnalysis.parse_yaml_raw(model_yaml)

    assert model_analysis == model_from_yaml == expected_model_analysis
