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

# noqa: F811

import pytest

from sparsezoo.utils import (
    extract_node_shapes_and_dtypes,
    get_layer_and_op_counts,
    get_node_bias,
    get_node_four_block_sparsity,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_sparsity,
    get_node_weight,
    get_num_dense_and_sparse_ops,
    get_zero_point,
    is_four_block_sparse_layer,
    is_parameterized_prunable_layer,
    is_quantized_layer,
    is_sparse_layer,
)
from tests.sparsezoo.analysis.helpers import (
    get_expected_analysis,
    get_model_and_node,
    get_model_onnx,
    get_test_model_names,
    model_paths,
)


@pytest.fixture()
def margin_of_error():
    return 0.05


@pytest.fixture(scope="session")
def get_model_node_shapes_and_dtypes(get_model_onnx, model_paths):
    model_node_shapes = {}
    model_node_dtypes = {}
    for model_name in model_paths.keys():
        model = get_model_onnx(model_name)
        node_shapes, node_dtypes = extract_node_shapes_and_dtypes(model)
        model_node_shapes[model_name] = node_shapes
        model_node_dtypes[model_name] = node_dtypes

    def _get_model_node_shapes_and_dtypes(model_name):
        return model_node_shapes[model_name], model_node_dtypes[model_name]

    return _get_model_node_shapes_and_dtypes


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ not in [
        "test_get_node_weight",
        "test_get_node_bias",
        "test_is_four_block_sparse_layer",
    ]:
        metafunc.parametrize("model_name", get_test_model_names())
        # metafunc.parametrize("model_name", ["mobilenet_v1_pruned_moderate"])


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
        ("resnet50_pruned_quantized", "DequantizeLinear_183", None),
        ("resnet50_pruned_quantized", "Conv_199_quant", (64, 256, 1, 1)),
        ("resnet50_pruned_quantized", "Add_254", None),
        ("resnet50_pruned_quantized", "QuantizeLinear_258", None),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1232", None),
        ("resnet50_pruned_quantized", "Gemm_1239", (2048, 1000)),
    ],
)
def test_get_node_weight(model_name, node_name, expected_shape, get_model_and_node):
    model, node = get_model_and_node(model_name, node_name)
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
        ("resnet50_pruned_quantized", "DequantizeLinear_183", None),
        ("resnet50_pruned_quantized", "Conv_199_quant", (64,)),
        ("resnet50_pruned_quantized", "Add_254", None),
        ("resnet50_pruned_quantized", "QuantizeLinear_258", None),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1232", None),
        ("resnet50_pruned_quantized", "Gemm_1239", (1000,)),
    ],
)
def test_get_node_bias(model_name, node_name, expected_shape, get_model_and_node):
    model, node = get_model_and_node(model_name, node_name)

    bias = get_node_bias(model, node)
    if expected_shape is None:
        assert bias is None
    else:
        assert bias is not None
        assert bias.shape == expected_shape


def test_is_parameterized_prunable_layer(
    model_name, get_expected_analysis, get_model_and_node
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        assert (
            is_parameterized_prunable_layer(model, node)
            == node_analysis.parameterized_and_prunable
        )


def test_get_layer_and_op_counts(model_name, get_model_onnx, get_expected_analysis):
    model = get_model_onnx(model_name)
    model_analysis = get_expected_analysis(model_name)

    layer_counts, op_counts = get_layer_and_op_counts(model)
    assert layer_counts == model_analysis.layer_counts
    assert op_counts == model_analysis.non_parameterized_operator_counts


def test_is_quantized_layer(model_name, get_model_and_node, get_expected_analysis):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        assert is_quantized_layer(model, node) == node_analysis.is_quantized_layer


def test_get_node_num_zeros_and_size(
    model_name,
    get_model_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        num_zeros, size = get_node_num_zeros_and_size(model, node)
        if node_analysis.weight is not None:
            assert num_zeros == node_analysis.weight.num_sparse_parameters
            assert size == node_analysis.weight.num_parameters
        else:
            assert num_zeros == 0
            assert size == 0


def test_get_node_num_four_block_zeros_and_size(
    model_name,
    get_model_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        num_zero_blocks, num_blocks = get_node_num_four_block_zeros_and_size(
            model, node
        )
        if node_analysis.weight is not None:
            assert num_zero_blocks == node_analysis.weight.num_sparse_four_blocks
            assert num_blocks == node_analysis.weight.num_four_blocks
        else:
            assert num_zero_blocks == 0
            assert num_blocks == 0


def test_get_zero_point(
    model_name,
    get_model_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        assert get_zero_point(model, node) == node_analysis.zero_point


def test_get_node_sparsity(
    model_name,
    get_model_and_node,
    get_expected_analysis,
    margin_of_error,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        if node_analysis.weight is not None:
            assert get_node_sparsity(model, node) == pytest.approx(
                node_analysis.weight.sparsity, abs=margin_of_error
            )
        else:
            assert get_node_sparsity(model, node) == pytest.approx(
                0, abs=margin_of_error
            )


def test_is_sparse_layer(
    model_name,
    get_model_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        assert is_sparse_layer(model, node) == node_analysis.is_sparse_layer


def test_get_node_four_block_sparsity(
    model_name,
    get_model_and_node,
    get_expected_analysis,
    margin_of_error,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        if node_analysis.weight is not None:
            assert get_node_four_block_sparsity(model, node) == pytest.approx(
                node_analysis.weight.four_block_sparsity, abs=margin_of_error
            )
        else:
            assert get_node_four_block_sparsity(model, node) == 0


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
        ("resnet50_pruned_quantized", "DequantizeLinear_183", False),
        ("resnet50_pruned_quantized", "Conv_199_quant", True),
        ("resnet50_pruned_quantized", "Add_254", False),
        ("resnet50_pruned_quantized", "QuantizeLinear_258", False),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1232", False),
        ("resnet50_pruned_quantized", "Gemm_1239", False),
    ],
)
def test_is_four_block_sparse_layer(
    model_name,
    node_name,
    expected_bool,
    get_model_and_node,
    margin_of_error,
):
    model, node = get_model_and_node(model_name, node_name)

    assert (
        is_four_block_sparse_layer(model, node, threshold=margin_of_error)
        == expected_bool
    )


def test_get_num_dense_and_sparse_ops(
    model_name,
    get_expected_analysis,
    get_model_and_node,
    get_model_node_shapes_and_dtypes,
    margin_of_error,
):
    model_analysis = get_expected_analysis(model_name)
    node_shapes, node_dtypes = get_model_node_shapes_and_dtypes(model_name)

    for node_analysis in model_analysis.nodes:
        model, node = get_model_and_node(model_name, node_analysis.name)
        num_dense_ops, num_sparse_ops = get_num_dense_and_sparse_ops(
            model, node, node_shapes=node_shapes
        )
        assert num_dense_ops == pytest.approx(
            node_analysis.num_dense_ops, abs=margin_of_error
        )
        assert num_sparse_ops == pytest.approx(
            node_analysis.num_sparse_ops, abs=margin_of_error
        )
