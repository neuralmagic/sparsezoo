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

import onnx
import pytest

from sparsezoo import Zoo
from sparsezoo.analysis.helpers import (
    get_layer_and_op_counts,
    get_node_bias,
    get_node_four_block_sparsity,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_sparsity,
    get_node_weight,
    get_num_operations,
    get_zero_point,
    is_four_block_sparse_layer,
    is_parameterized_prunable_layer,
    is_quantized_layer,
    is_sparse_layer,
)

from sparsezoo.analysis.onnx_helpers import (
    extract_node_shapes,
)


@pytest.fixture()
def margin_of_error():
    return 0.05


@pytest.fixture(scope="session")
def model_stubs():
    return {
        "yolact_none": "zoo:cv/segmentation/yolact-darknet53/"
        "pytorch/dbolya/coco/base-none",
        "mobilenet_v1_pruned_moderate": "zoo:cv/classification/mobilenet_v1-1.0/"
        "pytorch/sparseml/imagenet/pruned-moderate",
        "bert_pruned_quantized": "zoo:nlp/question_answering/bert-base/"
        "pytorch/huggingface/squad/"
        "12layer_pruned80_quant-none-vnni",
        "resnet50_pruned_quantized": "zoo:cv/classification/resnet_v1-50"
        "/pytorch/sparseml/imagenet/pruned85_quant-none-vnni",
        "resnet50_pruned85_vnni": "/Users/poketopa/Desktop/neuralmagic/models/resnet50_pruned85_vnni.onnx"
    }


@pytest.fixture(scope="session")
def get_model_onnx(model_stubs):
    model_onnxs = {}
    for model_name, model_stub in model_stubs.items():
        model_stub = model_stubs[model_name]
        if model_stub[:4] == "zoo:":
            model = Zoo.load_model_from_stub(model_stub)
            model.onnx_file.download()
            onnx_path = model.onnx_file.downloaded_path()
        else:
            onnx_path = model_stub
        print(onnx_path)
        model_onnx = onnx.load(onnx_path)
        model_onnxs[model_name] = model_onnx

    def _get_model_onnx(model_name):
        return model_onnxs[model_name]

    return _get_model_onnx


@pytest.fixture(scope="session")
def get_model_node_shapes(model_stubs, get_model_onnx):
    model_node_shapes = {}
    for model_name, model_stub in model_stubs.items():
        model = get_model_onnx(model_name)
        node_shapes = extract_node_shapes(model)
        model_node_shapes[model_name] = node_shapes

    def _get_model_node_shapes(model_name):
        return model_node_shapes[model_name]

    return _get_model_node_shapes


@pytest.fixture()
def get_node_from_name():
    def _get_node_from_name(model, node_name):
        return [node for node in list(model.graph.node) if node.name == node_name][0]

    return _get_node_from_name


@pytest.mark.parametrize(
    "model_name,node_name,expected_shape",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", (32, 3, 3, 3)),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", None),
        ("mobilenet_v1_pruned_moderate", "Pad_82", None),
        ("mobilenet_v1_pruned_moderate", "AveragePool_83", None),
        ("mobilenet_v1_pruned_moderate", "Shape_84", None),
        ("mobilenet_v1_pruned_moderate", "Gather_86", None),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", None),
        ("mobilenet_v1_pruned_moderate", "Concat_88", None),
        ("mobilenet_v1_pruned_moderate", "Reshape_89", None),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", (1024, 1000)),
        ("mobilenet_v1_pruned_moderate", "Softmax_91", None),
        ("bert_pruned_quantized", "Gather_34", (2, 768)),
        ("bert_pruned_quantized", "DequantizeLinear_27", None),
        ("bert_pruned_quantized", "MatMul_80_quant", (768, 768)),
        ("bert_pruned_quantized", "MatMul_157_quant", None),
        ("resnet50_pruned_quantized", "DequantizeLinear_22", None),
        ("resnet50_pruned_quantized", "Conv_431_quant", (512, 128, 1, 1)),
        ("resnet50_pruned_quantized", "Add_1168", None),
        ("resnet50_pruned_quantized", "QuantizeLinear_1178", None),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1328", None),
        ("resnet50_pruned_quantized", "Gemm_1335", (2048, 1000)),
    ],
)
def test_get_node_weight(
    model_name, node_name, expected_shape, get_model_onnx, get_node_from_name
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    weight = get_node_weight(model, node)
    if expected_shape is None:
        assert weight is None
    else:
        assert weight is not None
        assert weight.shape == expected_shape


@pytest.mark.parametrize(
    "model_name,node_name,expected_shape",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", None),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", None),
        ("mobilenet_v1_pruned_moderate", "Pad_82", None),
        ("mobilenet_v1_pruned_moderate", "AveragePool_83", None),
        ("mobilenet_v1_pruned_moderate", "Shape_84", None),
        ("mobilenet_v1_pruned_moderate", "Gather_86", None),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", None),
        ("mobilenet_v1_pruned_moderate", "Concat_88", None),
        ("mobilenet_v1_pruned_moderate", "Reshape_89", None),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", (1000,)),
        ("mobilenet_v1_pruned_moderate", "Softmax_91", None),
        ("bert_pruned_quantized", "Gather_34", None),
        ("bert_pruned_quantized", "DequantizeLinear_27", None),
        ("bert_pruned_quantized", "MatMul_80_quant", None),
        ("bert_pruned_quantized", "MatMul_157_quant", None),
        ("resnet50_pruned_quantized", "DequantizeLinear_22", None),
        ("resnet50_pruned_quantized", "Conv_431_quant", (512,)),
        ("resnet50_pruned_quantized", "Add_1168", None),
        ("resnet50_pruned_quantized", "QuantizeLinear_1178", None),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1328", None),
        ("resnet50_pruned_quantized", "Gemm_1335", (1000,)),
    ],
)
def test_get_node_bias(
    model_name, node_name, expected_shape, get_model_onnx, get_node_from_name
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    bias = get_node_bias(model, node)
    if expected_shape is None:
        assert bias is None
    else:
        assert bias is not None
        assert bias.shape == expected_shape


@pytest.mark.parametrize(
    "model_name,node_name,expected_bool",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", True),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", False),
        ("mobilenet_v1_pruned_moderate", "Pad_82", False),
        ("mobilenet_v1_pruned_moderate", "AveragePool_83", False),
        ("mobilenet_v1_pruned_moderate", "Shape_84", False),
        ("mobilenet_v1_pruned_moderate", "Gather_86", False),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", False),
        ("mobilenet_v1_pruned_moderate", "Concat_88", False),
        ("mobilenet_v1_pruned_moderate", "Reshape_89", False),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", True),
        ("mobilenet_v1_pruned_moderate", "Softmax_91", False),
        ("bert_pruned_quantized", "Gather_34", True),
        ("bert_pruned_quantized", "DequantizeLinear_27", False),
        ("bert_pruned_quantized", "MatMul_80_quant", True),
        ("bert_pruned_quantized", "MatMul_157_quant", False),
        ("resnet50_pruned_quantized", "Conv_431_quant", True),
        ("resnet50_pruned_quantized", "DequantizeLinear_22", False),
        ("resnet50_pruned_quantized", "Conv_431_quant", True),
        ("resnet50_pruned_quantized", "Add_1168", False),
        ("resnet50_pruned_quantized", "QuantizeLinear_1178", False),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1328", False),
        ("resnet50_pruned_quantized", "Gemm_1335", True),
        ("resnet50_pruned_quantized", "Softmax_1336", False),
    ],
)
def test_is_parameterized_prunable_layer(
    model_name, node_name, expected_bool, get_model_onnx, get_node_from_name
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert is_parameterized_prunable_layer(model, node) == expected_bool


@pytest.mark.parametrize(
    "model_name,expected_layer_counts,expected_op_counts",
    [
        (
            "mobilenet_v1_pruned_moderate",
            {"Conv": 27, "Gemm": 1},
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
                "MatMulInteger": 73,
                "Gather": 3,
            },
            {
                "Gather": 97,
                "Unsqueeze": 99,
                "Shape": 97,
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
                "Softmax": 12,
                "Erf": 12,
                "Split": 1,
                "Squeeze": 2,
                "QLinearMatMul": 24,
            },
        ),
    ],
)
def test_get_layer_and_op_counts(
    model_name, expected_layer_counts, expected_op_counts, get_model_onnx
):
    model = get_model_onnx(model_name)

    layer_counts, op_counts = get_layer_and_op_counts(model)
    assert layer_counts == expected_layer_counts
    assert op_counts == expected_op_counts


@pytest.mark.parametrize(
    "model_name,node_name,expected_bool",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_72", False),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", False),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", False),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", False),
        ("bert_pruned_quantized", "Gather_34", True),
        ("bert_pruned_quantized", "DequantizeLinear_27", False),
        ("bert_pruned_quantized", "MatMul_80_quant", True),
        ("bert_pruned_quantized", "MatMul_157_quant", True),
        ("resnet50_pruned_quantized", "Conv_431_quant", True),
        ("resnet50_pruned_quantized", "Gemm_1335", False),
    ],
)
def test_is_quantized_layer(
    model_name, node_name, expected_bool, get_model_onnx, get_node_from_name
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert is_quantized_layer(model, node) == expected_bool


@pytest.mark.parametrize(
    "model_name,node_name,expected_num_zeros,expected_param_size",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_72", 471859, 524288),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", 0, 0),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", 0, 0),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", 0, 1024000),
        ("bert_pruned_quantized", "Gather_34", 0, 1536),
        ("bert_pruned_quantized", "DequantizeLinear_27", 0, 0),
        ("bert_pruned_quantized", "MatMul_80_quant", 473571, 589824),
        ("bert_pruned_quantized", "MatMul_157_quant", 0, 0),
        ("resnet50_pruned_quantized", "Conv_431_quant", 57451, 65536),
        ("resnet50_pruned_quantized", "Gemm_1335", 0, 2048000),
    ],
)
def test_get_node_num_zeros_and_size(
    model_name,
    node_name,
    expected_num_zeros,
    expected_param_size,
    get_model_onnx,
    get_node_from_name,
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert get_node_num_zeros_and_size(model, node) == (
        expected_num_zeros,
        expected_param_size,
    )


@pytest.mark.parametrize(
    "model_name,node_name,expected_num_zero_blocks,expected_num_blocks",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_72", 93329, 131072),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", 0, 0),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", 0, 0),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", 0, 256000),
        ("bert_pruned_quantized", "Gather_34", 0, 768),
        ("bert_pruned_quantized", "DequantizeLinear_27", 0, 0),
        ("bert_pruned_quantized", "MatMul_80_quant", 117964, 147456),
        ("bert_pruned_quantized", "MatMul_157_quant", 0, 0),
        ("resnet50_pruned_quantized", "Conv_431_quant", 13938, 16384),
        ("resnet50_pruned_quantized", "Gemm_1335", 0, 512000),
    ],
)
def test_get_node_num_four_block_zeros_and_size(
    model_name,
    node_name,
    expected_num_zero_blocks,
    expected_num_blocks,
    get_model_onnx,
    get_node_from_name,
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert get_node_num_four_block_zeros_and_size(model, node) == (
        expected_num_zero_blocks,
        expected_num_blocks,
    )


@pytest.mark.parametrize(
    "model_name,node_name,expected_value",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_72", 0),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", 0),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", 0),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", 0),
        ("bert_pruned_quantized", "Gather_34", 0),
        ("bert_pruned_quantized", "DequantizeLinear_27", 0),
        ("bert_pruned_quantized", "MatMul_80_quant", 128),
        ("bert_pruned_quantized", "MatMul_157_quant", 115),
        ("resnet50_pruned_quantized", "Conv_431_quant", 128),
        ("resnet50_pruned_quantized", "Gemm_1335", 0),
    ],
)
def test_get_zero_point(
    model_name, node_name, expected_value, get_model_onnx, get_node_from_name
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert get_zero_point(model, node) == expected_value


@pytest.mark.parametrize(
    "model_name,node_name,expected_value",
    [
        ("yolact_none", "Conv_0", 0),
        ("yolact_none", "Conv_33", 0),
        ("yolact_none", "LeakyRelu_36", 0),
        ("yolact_none", "Conv_275", 0),
        ("mobilenet_v1_pruned_moderate", "Conv_72", 0.90),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", 0),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", 0),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", 0),
        ("bert_pruned_quantized", "Gather_34", 0),
        ("bert_pruned_quantized", "DequantizeLinear_27", 0),
        ("bert_pruned_quantized", "MatMul_80_quant", 0.80),
        ("bert_pruned_quantized", "MatMul_157_quant", 0),
        ("resnet50_pruned_quantized", "Conv_431_quant", 0.85),
        ("resnet50_pruned_quantized", "Gemm_1335", 0),
    ],
)
def test_get_node_sparsity(
    model_name,
    node_name,
    expected_value,
    get_model_onnx,
    get_node_from_name,
    margin_of_error,
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert get_node_sparsity(model, node) == pytest.approx(
        expected_value, abs=margin_of_error
    )


@pytest.mark.parametrize(
    "model_name,node_name,expected_bool",
    [
        ("yolact_none", "Conv_0", False),
        ("yolact_none", "Conv_33", False),
        ("yolact_none", "LeakyRelu_36", False),
        ("yolact_none", "Conv_275", False),
        ("mobilenet_v1_pruned_moderate", "Conv_72", True),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", False),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", False),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", False),
        ("bert_pruned_quantized", "Gather_34", False),
        ("bert_pruned_quantized", "DequantizeLinear_27", False),
        ("bert_pruned_quantized", "MatMul_80_quant", True),
        ("bert_pruned_quantized", "MatMul_157_quant", False),
        ("resnet50_pruned_quantized", "Conv_431_quant", True),
        ("resnet50_pruned_quantized", "Gemm_1335", False),
    ],
)
def test_is_sparse_layer(
    model_name, node_name, expected_bool, get_model_onnx, get_node_from_name
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert is_sparse_layer(model, node) == expected_bool


@pytest.mark.parametrize(
    "model_name,node_name,expected_value",
    [
        ("yolact_none", "Conv_0", 0),
        ("yolact_none", "Conv_33", 0),
        ("yolact_none", "LeakyRelu_36", 0),
        ("yolact_none", "Conv_275", 0),
        ("mobilenet_v1_pruned_moderate", "Conv_57", 0),
        ("mobilenet_v1_pruned_moderate", "Conv_72", 0.67),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", 0),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", 0),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", 0),
        ("bert_pruned_quantized", "Gather_34", 0),
        ("bert_pruned_quantized", "DequantizeLinear_27", 0),
        ("bert_pruned_quantized", "MatMul_238_quant", 0.80),
        ("bert_pruned_quantized", "MatMul_446_quant", 0.80),
        ("bert_pruned_quantized", "MatMul_157_quant", 0),
        ("resnet50_pruned_quantized", "Conv_211_quant", 0.85),
        ("resnet50_pruned_quantized", "Conv_408_quant", 0.85),
        ("resnet50_pruned_quantized", "Conv_431_quant", 0.85),
        ("resnet50_pruned_quantized", "Conv_1311_quant", 0.85),
        ("resnet50_pruned_quantized", "Gemm_1335", 0),
    ],
)
def test_get_node_four_block_sparsity(
    model_name,
    node_name,
    expected_value,
    get_model_onnx,
    get_node_from_name,
    margin_of_error,
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)
    assert get_node_four_block_sparsity(model, node) == pytest.approx(
        expected_value, abs=margin_of_error
    )


@pytest.mark.parametrize(
    "model_name,node_name,expected_bool",
    [
        ("yolact_none", "Conv_0", False),
        ("yolact_none", "Conv_33", False),
        ("yolact_none", "LeakyRelu_36", False),
        ("yolact_none", "Conv_275", False),
        ("mobilenet_v1_pruned_moderate", "Conv_57", False),
        ("mobilenet_v1_pruned_moderate", "Conv_72", False),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", False),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", False),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", False),
        ("bert_pruned_quantized", "Gather_34", False),
        ("bert_pruned_quantized", "DequantizeLinear_27", False),
        ("bert_pruned_quantized", "MatMul_238_quant", True),
        ("bert_pruned_quantized", "MatMul_157_quant", False),
        ("resnet50_pruned_quantized", "Conv_13_quant", False),
        ("resnet50_pruned_quantized", "Conv_211_quant", True),
        ("resnet50_pruned_quantized", "Conv_408_quant", True),
        ("resnet50_pruned_quantized", "Conv_431_quant", True),
        ("resnet50_pruned_quantized", "Conv_1311_quant", True),
        ("resnet50_pruned_quantized", "Gemm_1335", 0),
    ],
)
def test_is_four_block_sparse_layer(
    model_name,
    node_name,
    expected_bool,
    get_model_onnx,
    get_node_from_name,
    margin_of_error,
):
    model = get_model_onnx(model_name)
    node = get_node_from_name(model, node_name)

    assert (
        is_four_block_sparse_layer(model, node, threshold=margin_of_error)
        == expected_bool
    )


@pytest.mark.parametrize(
    "model_name,node_name,expected_value",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", 10838016),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", 401408),
        ("mobilenet_v1_pruned_moderate", "Pad_82", 0),
        ("mobilenet_v1_pruned_moderate", "AveragePool_83", 50176),
        ("mobilenet_v1_pruned_moderate", "Shape_84", 0),
        ("mobilenet_v1_pruned_moderate", "Gather_86", 0),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", 0),
        ("mobilenet_v1_pruned_moderate", "Concat_88", 0),
        ("mobilenet_v1_pruned_moderate", "Reshape_89", 0),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", 2049000),
        ("mobilenet_v1_pruned_moderate", "Softmax_91", 0),
        ("bert_pruned_quantized", "Gather_34", 0),
        ("bert_pruned_quantized", "DequantizeLinear_27", 0),
        ("bert_pruned_quantized", "MatMul_80_quant", 1179648),
        ("bert_pruned_quantized", "MatMul_157_quant", 224722944),
        ("resnet50_pruned_quantized", "Conv_431_quant", 51380736),
        ("resnet50_pruned_quantized", "DequantizeLinear_22", 0),
        ("resnet50_pruned_quantized", "Add_1168", 100352),
        ("resnet50_pruned_quantized", "QuantizeLinear_1178", 0),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1328", 100352),
        ("resnet50_pruned_quantized", "Gemm_1335", 4097000),
        ("resnet50_pruned_quantized", "Softmax_1336", 0),
    ],
)
def test_get_num_operations(
    model_name, node_name, expected_value, get_model_onnx, get_node_from_name, get_model_node_shapes
):
    model = get_model_onnx(model_name)
    node_shapes = get_model_node_shapes(model_name)
    node = get_node_from_name(model, node_name)

    assert get_num_operations(model, node, node_shapes=node_shapes) == expected_value





from sparsezoo.analysis.ops_calcs import (
    get_num_dense_and_sparse_ops,
)


@pytest.mark.parametrize(
    "model_name,node_name,expected_value",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", (21676032, 0)),
        ("mobilenet_v1_pruned_moderate", "Conv_27", (14112, 0)),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", (401408, 0)),
        #("mobilenet_v1_pruned_moderate", "Pad_82", 0),
        #("mobilenet_v1_pruned_moderate", "AveragePool_83", 50176),
        #("mobilenet_v1_pruned_moderate", "Shape_84", 0),
        #("mobilenet_v1_pruned_moderate", "Gather_86", 0),
        #("mobilenet_v1_pruned_moderate", "Unsqueeze_87", 0),
        #("mobilenet_v1_pruned_moderate", "Concat_88", 0),
        #("mobilenet_v1_pruned_moderate", "Reshape_89", 0),

        ("mobilenet_v1_pruned_moderate", "Gemm_90", (2048000, 0)),
        ("resnet50_pruned85_vnni", "Gemm_1335", (4096000, 0)),
        ("resnet50_pruned85_vnni", "Conv_158_quant", (35073024, 196539392)),

        #("mobilenet_v1_pruned_moderate", "Softmax_91", 0),
        #("bert_pruned_quantized", "Gather_34", 0),
        #("bert_pruned_quantized", "DequantizeLinear_27", 0),
        ("bert_pruned_quantized", "MatMul_80_quant", (235936, 943712)),
        ("bert_pruned_quantized", "MatMul_157_quant", (224722944, 0)),
        ("resnet50_pruned_quantized", "Conv_431_quant", (51380736, 0)),
        #("resnet50_pruned_quantized", "DequantizeLinear_22", 0),
        #("resnet50_pruned_quantized", "Add_1168", 100352),
        #("resnet50_pruned_quantized", "QuantizeLinear_1178", 0),
        #("resnet50_pruned_quantized", "GlobalAveragePool_1328", 100352),
        ("resnet50_pruned_quantized", "Gemm_1335", (4098000, 0)),
        #("resnet50_pruned_quantized", "Softmax_1336", 0),
    ],
)
def test_get_num_dense_and_sparse_ops(
    model_name, node_name, expected_value, get_model_onnx, get_node_from_name, get_model_node_shapes
):
    model = get_model_onnx(model_name)
    node_shapes = get_model_node_shapes(model_name)
    node = get_node_from_name(model, node_name)

    print(get_num_operations(model, node, node_shapes=node_shapes))
    assert get_num_dense_and_sparse_ops(model, node, node_shapes=node_shapes) == expected_value
