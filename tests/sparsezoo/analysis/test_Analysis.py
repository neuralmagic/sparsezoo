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
from sparsezoo.analysis import ModelAnalysis, NodeAnalysis


@pytest.fixture()
def margin_of_error():
    return 0.05


@pytest.fixture(scope="session")
def get_model_analysis():
    model_stubs = {
        "yolact_none": "zoo:cv/segmentation/yolact-darknet53/"
        "pytorch/dbolya/coco/base-none",
        "mobilenet_v1_pruned_moderate": "zoo:cv/classification/mobilenet_v1-1.0/"
        "pytorch/sparseml/imagenet/pruned-moderate",
        "bert_pruned_quantized": "zoo:nlp/question_answering/bert-base/"
        "pytorch/huggingface/squad/"
        "12layer_pruned80_quant-none-vnni",
        "resnet50_pruned_quantized": "zoo:cv/classification/resnet_v1-50"
        "/pytorch/sparseml/imagenet/pruned85_quant-none-vnni",
    }

    model_analyses = {}
    for model_name, model_stub in model_stubs.items():
        model_stub = model_stubs[model_name]
        model = Zoo.load_model_from_stub(model_stub)
        model.onnx_file.download()
        onnx_path = model.onnx_file.downloaded_path()
        analysis = ModelAnalysis.from_onnx_model(onnx_path)
        model_analyses[model_name] = analysis

    def _get_model_analysis(model_name):
        return model_analyses[model_name]

    return _get_model_analysis


@pytest.mark.parametrize(
    "model_name,expected_dict",
    [
        ("yolact_none", {"Conv": 85}),
        ("mobilenet_v1_pruned_moderate", {"Conv": 27, "Gemm": 1}),
        ("bert_pruned_quantized", {"Gather": 3, "MatMulInteger": 73}),
        ("resnet50_pruned_quantized", {"Gemm": 1, "QLinearConv": 53}),
    ],
)
def test_layer_counts(model_name, expected_dict, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.layer_counts == expected_dict


@pytest.mark.parametrize(
    "model_name,expected_dict",
    [
        (
            "yolact_none",
            {
                "LeakyRelu": 52,
                "Add": 26,
                "Constant": 32,
                "Shape": 21,
                "Gather": 19,
                "Unsqueeze": 19,
                "Concat": 22,
                "Slice": 2,
                "Cast": 2,
                "Resize": 3,
                "Relu": 14,
                "Transpose": 16,
                "Reshape": 15,
                "Tanh": 5,
                "Softmax": 1,
            },
        ),
        (
            "mobilenet_v1_pruned_moderate",
            {
                "BatchNormalization": 27,
                "Relu": 27,
                "Constant": 2,
                "Pad": 1,
                "AveragePool": 1,
                "Shape": 1,
                "Gather": 1,
                "Unsqueeze": 1,
                "Concat": 1,
                "Reshape": 1,
                "Softmax": 1,
            },
        ),
        (
            "bert_pruned_quantized",
            {
                "Unsqueeze": 99,
                "Shape": 97,
                "Gather": 97,
                "DequantizeLinear": 39,
                "Cast": 74,
                "Add": 174,
                "Sub": 26,
                "Mul": 123,
                "Slice": 1,
                "ReduceMean": 50,
                "Pow": 25,
                "Sqrt": 25,
                "Div": 49,
                "QuantizeLinear": 97,
                "Concat": 48,
                "Reshape": 48,
                "Transpose": 48,
                "QLinearMatMul": 24,
                "Softmax": 12,
                "Erf": 12,
                "Split": 1,
                "Squeeze": 2,
            },
        ),
        (
            "resnet50_pruned_quantized",
            {
                "QuantizeLinear": 17,
                "DequantizeLinear": 33,
                "MaxPool": 1,
                "Add": 16,
                "GlobalAveragePool": 1,
                "Shape": 1,
                "Gather": 1,
                "Unsqueeze": 1,
                "Concat": 1,
                "Reshape": 1,
                "Softmax": 1,
            },
        ),
    ],
)
def test_op_counts(model_name, expected_dict, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.op_counts == expected_dict


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 334),
        ("mobilenet_v1_pruned_moderate", 92),
        ("bert_pruned_quantized", 1247),
        ("resnet50_pruned_quantized", 128),
    ],
)
def test_num_nodes(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_nodes == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 85),
        ("mobilenet_v1_pruned_moderate", 28),
        ("bert_pruned_quantized", 76),
        ("resnet50_pruned_quantized", 54),
    ],
)
def test_num_layers(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_layers == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 249),
        ("mobilenet_v1_pruned_moderate", 64),
        ("bert_pruned_quantized", 1171),
        ("resnet50_pruned_quantized", 74),
    ],
)
def test_num_operations(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_operations == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0),
        ("mobilenet_v1_pruned_moderate", 0.65),
        ("bert_pruned_quantized", 0.63),
        ("resnet50_pruned_quantized", 0.79),
    ],
)
def test_average_sparsity(
    model_name, expected_value, get_model_analysis, margin_of_error
):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.average_sparsity == pytest.approx(
        expected_value, abs=margin_of_error
    )


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0.0),
        ("mobilenet_v1_pruned_moderate", 0.45),
        ("bert_pruned_quantized", 0.62),
        ("resnet50_pruned_quantized", 0.78),
    ],
)
def test_average_four_block_sparsity(
    model_name, expected_value, get_model_analysis, margin_of_error
):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.average_four_block_sparsity == pytest.approx(
        expected_value, abs=margin_of_error
    )


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0),
        ("mobilenet_v1_pruned_moderate", 12),
        ("bert_pruned_quantized", 75),
        ("resnet50_pruned_quantized", 53),
    ],
)
def test_num_sparse_layers(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_sparse_layers == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0),
        ("mobilenet_v1_pruned_moderate", 0),
        ("bert_pruned_quantized", 100),
        ("resnet50_pruned_quantized", 53),
    ],
)
def test_num_quantized_layers(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_quantized_layers == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 53317216),
        ("mobilenet_v1_pruned_moderate", 4209088),
        ("bert_pruned_quantized", 108771840),
        ("resnet50_pruned_quantized", 25502912),
    ],
)
def test_num_parameters(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_parameters == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0),
        ("mobilenet_v1_pruned_moderate", 2740224),
        ("bert_pruned_quantized", 68923743),
        ("resnet50_pruned_quantized", 20203056),
    ],
)
def test_num_sparse_parameters(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_sparse_parameters == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 13329376),
        ("mobilenet_v1_pruned_moderate", 1085824),
        ("bert_pruned_quantized", 27193728),
        ("resnet50_pruned_quantized", 6376512),
    ],
)
def test_num_four_blocks(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_four_blocks == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0),
        ("mobilenet_v1_pruned_moderate", 487183),
        ("bert_pruned_quantized", 16986888),
        ("resnet50_pruned_quantized", 4982870),
    ],
)
def test_num_sparse_four_blocks(model_name, expected_value, get_model_analysis):
    model_analysis = get_model_analysis(model_name)

    assert model_analysis.num_sparse_four_blocks == expected_value


@pytest.mark.parametrize(
    "model_name,expected_node_analysis",
    [
        (
            "yolact_none",
            NodeAnalysis(
                name="Conv_0",
                sparsity=0.0,
                four_block_sparsity=0.0,
                param_size=864,
                num_sparse_values=0,
                num_four_blocks=288,
                num_sparse_four_blocks=0,
                quantized_layer=False,
                zero_point=0,
                dtype="float32",
            ),
        ),
        (
            "mobilenet_v1_pruned_moderate",
            NodeAnalysis(
                name="Conv_72",
                sparsity=0.8999996185302734,
                four_block_sparsity=0.7120437622070312,
                param_size=524288,
                num_sparse_values=471859,
                num_four_blocks=131072,
                num_sparse_four_blocks=93329,
                quantized_layer=False,
                zero_point=0,
                dtype="float32",
            ),
        ),
        (
            "bert_pruned_quantized",
            NodeAnalysis(
                name="MatMul_80_quant",
                sparsity=0.8029022216796875,
                four_block_sparsity=0.7999945746527778,
                param_size=589824,
                num_sparse_values=473571,
                num_four_blocks=147456,
                num_sparse_four_blocks=117964,
                quantized_layer=True,
                zero_point=128,
                dtype="uint8",
            ),
        ),
    ],
)
def test_node_analyses(model_name, expected_node_analysis, get_model_analysis):
    model_analysis = get_model_analysis(model_name)
    nodes = model_analysis.nodes
    found_nodes = [node for node in nodes if node.name == expected_node_analysis.name]
    assert len(found_nodes) == 1
    found_node = found_nodes[0]

    assert found_node == expected_node_analysis
