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

# flake8: noqa

import pytest

from sparsezoo.utils import (
    extract_node_id,
    extract_node_shapes_and_dtypes,
    get_layer_and_op_counts,
    get_node_bias,
    get_node_bias_name,
    get_node_four_block_sparsity,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_sparsity,
    get_node_weight,
    get_node_weight_name,
    get_ops_dict,
    get_zero_point,
    is_four_block_sparse_layer,
    is_parameterized_prunable_layer,
    is_quantized_layer,
    is_sparse_layer,
)
from tests.sparsezoo.analyze.helpers import (
    get_expected_analysis,
    get_generated_analysis,
    get_model_graph,
    get_model_graph_and_node,
    get_test_model_names,
    model_paths,
)


@pytest.fixture()
def margin_of_error():
    return 0.05


@pytest.fixture(scope="session")
def get_model_node_shapes_and_dtypes(get_model_graph, model_paths):
    model_node_shapes = {}
    model_node_dtypes = {}
    for model_name in model_paths.keys():
        model_graph = get_model_graph(model_name)
        node_shapes, node_dtypes = extract_node_shapes_and_dtypes(model_graph.model)
        model_node_shapes[model_name] = node_shapes
        model_node_dtypes[model_name] = node_dtypes

    def _get_model_node_shapes_and_dtypes(model_name):
        return model_node_shapes[model_name], model_node_dtypes[model_name]

    return _get_model_node_shapes_and_dtypes


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ not in [
        "test_get_node_weight_and_shape",
        "test_get_node_bias",
        "test_is_four_block_sparse_layer",
        "test_get_node_weight_name",
        "test_get_node_bias_name",
    ]:
        metafunc.parametrize("model_name", get_test_model_names())


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
def test_get_node_weight_and_shape(
    model_name, node_name, expected_shape, get_model_graph_and_node
):
    model_graph, node = get_model_graph_and_node(model_name, node_name)
    weight = get_node_weight(model_graph, node)
    weight_shape = weight.shape if weight is not None else None
    if expected_shape is None:
        assert weight is None
        assert weight_shape is None
    else:
        assert weight is not None
        assert weight.shape == expected_shape
        assert weight_shape is not None
        assert weight_shape == expected_shape


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
def test_get_node_bias(model_name, node_name, expected_shape, get_model_graph_and_node):
    model_graph, node = get_model_graph_and_node(model_name, node_name)

    bias = get_node_bias(model_graph, node)
    if expected_shape is None:
        assert bias is None
    else:
        assert bias is not None
        assert bias.shape == expected_shape


def test_is_parameterized_prunable_layer(
    model_name, get_expected_analysis, get_model_graph_and_node
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)
        assert (
            is_parameterized_prunable_layer(model_graph, node)
            == node_analysis.parameterized_prunable
        )


def test_get_layer_and_op_counts(model_name, get_model_graph, get_expected_analysis):
    model_graph = get_model_graph(model_name)
    model_analysis = get_expected_analysis(model_name)

    layer_counts, op_counts = get_layer_and_op_counts(model_graph)
    node_counts = layer_counts.copy()
    node_counts.update(op_counts)
    assert node_counts == model_analysis.node_counts


def test_is_quantized_layer(
    model_name, get_model_graph_and_node, get_expected_analysis
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)
        assert is_quantized_layer(model_graph, node) == node_analysis.quantized_node


def test_get_node_num_zeros_and_size(
    model_name,
    get_model_graph_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)
        num_zeros, size = get_node_num_zeros_and_size(model_graph, node)

        weight_analysis = next(
            (
                parameter
                for parameter in node_analysis.parameters
                if parameter.alias == "weight"
            ),
            None,
        )
        if weight_analysis is not None:
            assert size == weight_analysis.parameter_summary.total
            assert num_zeros == weight_analysis.parameter_summary.pruned
        else:
            assert size == 0
            assert num_zeros == 0


def test_get_node_num_four_block_zeros_and_size(
    model_name,
    get_model_graph_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)
        num_zero_blocks, num_blocks = get_node_num_four_block_zeros_and_size(
            model_graph, node
        )
        num_non_zero_blocks = num_blocks - num_zero_blocks

        weight_analysis = next(
            (
                parameter
                for parameter in node_analysis.parameters
                if parameter.alias == "weight"
            ),
            None,
        )
        if weight_analysis is not None:
            assert (
                num_non_zero_blocks
                == weight_analysis.parameter_summary.block_structure["block4"].non_zero
            )
            assert (
                num_zero_blocks
                == weight_analysis.parameter_summary.block_structure["block4"].zero
            )
        else:
            assert num_non_zero_blocks == 0
            assert num_zero_blocks == 0


def test_get_zero_point(
    model_name,
    get_model_graph_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)
        assert get_zero_point(model_graph, node) == node_analysis.zero_point


def test_get_node_sparsity(
    model_name,
    get_model_graph_and_node,
    get_expected_analysis,
    margin_of_error,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)

        weight_analysis = next(
            (
                parameter
                for parameter in node_analysis.parameters
                if parameter.alias == "weight"
            ),
            None,
        )
        if weight_analysis is not None:
            assert get_node_sparsity(model_graph, node) == pytest.approx(
                weight_analysis.parameter_summary.block_structure["single"].sparsity,
                abs=margin_of_error,
            )
        else:
            assert get_node_sparsity(model_graph, node) == pytest.approx(
                0, abs=margin_of_error
            )


def test_is_sparse_layer(
    model_name,
    get_model_graph_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)
        assert is_sparse_layer(model_graph, node) == node_analysis.sparse_node


def test_get_node_four_block_sparsity(
    model_name,
    get_model_graph_and_node,
    get_expected_analysis,
    margin_of_error,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)

        weight_analysis = next(
            (
                parameter
                for parameter in node_analysis.parameters
                if parameter.alias == "weight"
            ),
            None,
        )
        if weight_analysis is not None:
            assert get_node_four_block_sparsity(model_graph, node) == pytest.approx(
                weight_analysis.parameter_summary.block_structure["block4"].sparsity,
                abs=margin_of_error,
            )
        else:
            assert get_node_four_block_sparsity(model_graph, node) == 0


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
    get_model_graph_and_node,
    margin_of_error,
):
    model_graph, node = get_model_graph_and_node(model_name, node_name)

    assert (
        is_four_block_sparse_layer(model_graph, node, threshold=margin_of_error)
        == expected_bool
    )


@pytest.mark.parametrize(
    "model_name,node_name,expected_name",
    [
        ("yolact_none", "Conv_0", "799"),
        ("yolact_none", "Conv_33", "841"),
        ("yolact_none", "LeakyRelu_36", None),
        ("yolact_none", "Conv_275", "prediction_layers.0.upfeature.0.weight"),
        ("mobilenet_v1_pruned_moderate", "Conv_57", "sections.3.4.depth.conv.weight"),
        ("mobilenet_v1_pruned_moderate", "Conv_72", "sections.4.0.point.conv.weight"),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", None),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", None),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", "classifier.fc.weight"),
        (
            "bert_pruned_quantized",
            "Gather_34",
            "bert.embeddings.token_type_embeddings.weight_quant",
        ),
        ("bert_pruned_quantized", "DequantizeLinear_27", None),
        ("bert_pruned_quantized", "MatMul_238_quant", "MatMul_238.weight_quantized"),
        ("bert_pruned_quantized", "MatMul_157_quant", None),
        ("resnet50_pruned_quantized", "DequantizeLinear_183", None),
        ("resnet50_pruned_quantized", "Conv_199_quant", "Conv_199.weight_quantized"),
        ("resnet50_pruned_quantized", "Add_254", None),
        ("resnet50_pruned_quantized", "QuantizeLinear_258", None),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1232", None),
        ("resnet50_pruned_quantized", "Gemm_1239", "classifier.fc.weight"),
    ],
)
def test_get_node_weight_name(
    model_name,
    node_name,
    expected_name,
    get_model_graph_and_node,
    margin_of_error,
):
    model_graph, node = get_model_graph_and_node(model_name, node_name)

    weight_name = get_node_weight_name(model_graph, node)
    if expected_name is None:
        assert weight_name is None
    else:
        assert weight_name == expected_name


@pytest.mark.parametrize(
    "model_name,node_name,expected_name",
    [
        ("yolact_none", "Conv_0", "800"),
        ("yolact_none", "Conv_33", "842"),
        ("yolact_none", "LeakyRelu_36", None),
        ("yolact_none", "Conv_275", "prediction_layers.0.upfeature.0.bias"),
        ("mobilenet_v1_pruned_moderate", "Conv_57", None),
        ("mobilenet_v1_pruned_moderate", "Conv_72", None),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", None),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", None),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", "classifier.fc.bias"),
        (
            "bert_pruned_quantized",
            "Gather_34",
            None,
        ),
        ("bert_pruned_quantized", "DequantizeLinear_27", None),
        ("bert_pruned_quantized", "MatMul_238_quant", None),
        ("bert_pruned_quantized", "MatMul_157_quant", None),
        ("resnet50_pruned_quantized", "DequantizeLinear_183", None),
        ("resnet50_pruned_quantized", "Conv_199_quant", "Conv_199.bias_quantized"),
        ("resnet50_pruned_quantized", "Add_254", None),
        ("resnet50_pruned_quantized", "QuantizeLinear_258", None),
        ("resnet50_pruned_quantized", "GlobalAveragePool_1232", None),
        ("resnet50_pruned_quantized", "Gemm_1239", "classifier.fc.bias"),
    ],
)
def test_get_node_bias_name(
    model_name,
    node_name,
    expected_name,
    get_model_graph_and_node,
    margin_of_error,
):
    model_graph, node = get_model_graph_and_node(model_name, node_name)

    bias_name = get_node_bias_name(node)
    if expected_name is None:
        assert bias_name is None
    else:
        assert bias_name == expected_name


def test_extract_node_id(
    model_name,
    get_model_graph_and_node,
    get_expected_analysis,
):
    model_analysis = get_expected_analysis(model_name)
    for node_analysis in model_analysis.nodes:
        _, node = get_model_graph_and_node(model_name, node_analysis.name)
        assert extract_node_id(node) == node_analysis.node_id


def test_get_ops_dict(
    model_name,
    get_expected_analysis,
    get_model_graph_and_node,
    get_model_node_shapes_and_dtypes,
    margin_of_error,
):
    model_analysis = get_expected_analysis(model_name)
    node_shapes, node_dtypes = get_model_node_shapes_and_dtypes(model_name)

    for node_analysis in model_analysis.nodes:
        model_graph, node = get_model_graph_and_node(model_name, node_analysis.name)
        node_shape = node_shapes.get(node_analysis.node_id)
        ops_dict = get_ops_dict(model_graph, node, node_shape=node_shape)

        def _sum_across_keys(dict, key):
            return sum([dict[k][key] for k in dict.keys()])

        dense_ops = _sum_across_keys(ops_dict, "num_dense_ops")
        pruned_ops = _sum_across_keys(ops_dict, "num_sparse_ops")

        assert dense_ops + pruned_ops == pytest.approx(
            node_analysis.operation_summary.ops.total, abs=margin_of_error
        )
        assert pruned_ops == pytest.approx(
            node_analysis.operation_summary.ops.pruned, abs=margin_of_error
        )
