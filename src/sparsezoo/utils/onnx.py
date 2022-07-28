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

from typing import Any, Dict, Optional, Tuple, Union

import numpy
from onnx import NodeProto, numpy_helper
from onnx.helper import get_attribute_value

from sparsezoo.utils import ONNXGraph


__all__ = [
    "get_layer_and_op_counts",
    "get_node_four_block_sparsity",
    "get_node_num_four_block_zeros_and_size",
    "get_node_sparsity",
    "get_node_weight_name",
    "get_node_weight",
    "get_node_bias",
    "get_node_bias_name",
    "get_node_num_zeros_and_size",
    "get_zero_point",
    "is_four_block_sparse_layer",
    "is_parameterized_prunable_layer",
    "is_quantized_layer",
    "is_sparse_layer",
    "group_four_block",
    "extract_node_id",
    "get_node_attributes",
]


def get_node_attributes(node: NodeProto) -> Dict[str, Any]:
    """
    :param node: node to which the attributes belong to
    :return: a dictionary of attribute name and value pairs
    """
    attributes = {}
    for attribute in node.attribute:
        attributes[attribute.name] = get_attribute_value(attribute)

    return attributes


def get_node_bias_name(node: NodeProto) -> str:
    """
    :param node: node potentially containing a bias
    :return: the name of the bias of the node
    """
    if node.op_type in ["Conv", "Gemm"]:
        return _get_node_input(node, 2, default=None)

    if node.op_type == "QLinearConv":
        return _get_node_input(node, 8, default=None)

    return None


def get_node_bias(model_graph: ONNXGraph, node: NodeProto) -> numpy.ndarray:
    """
    Finds parameter value of node (the node weight)

    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node to which parameter belongs to
    :return: a numpy array of param value, None if not found
    """

    initializer_name = get_node_bias_name(node)
    return get_initializer_value(model_graph, node, initializer_name)


def get_initializer_value(
    model_graph: ONNXGraph, node: NodeProto, initializer_name: Union[str, None]
) -> Union[numpy.ndarray, None]:
    """
    Helper function to find initializers by name in model graph
    :param model_graph: model graph that contains the initializer with the given name
    :param node: node to which the initializer belongs to
    :param initializer_name: name of initializer being returned
    :return: initalizer if found, None otherwise
    """
    if initializer_name is None:
        return None

    def _is_transposed_initializer(node: NodeProto, initailizer_name: str) -> bool:
        if node.op_type in ["Gemm"]:
            input_index = list(node.input).index(initailizer_name)
            node_attributes = get_node_attributes(node)

            if (
                input_index == 0
                and "transA" in node_attributes
                and node_attributes["transA"] == 1
            ):
                return True
            if (
                input_index == 1
                and "transB" in node_attributes
                and node_attributes["transB"] == 1
            ):
                return True

        return False

    initializer = model_graph.get_init_by_name(initializer_name)
    if initializer is None:
        return None

    value = numpy_helper.to_array(initializer)
    if _is_transposed_initializer(node, initializer_name):
        value = value.T

    return value


def get_node_num_zeros_and_size(
    model_graph: ONNXGraph, node: NodeProto
) -> Tuple[int, int]:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node whose number of zeros and parameter size is being calculated
    :return: number of zeros and number of total values in node parameter
    """
    zero_point = get_zero_point(model_graph, node)
    weight = get_node_weight(model_graph, node)
    if weight is None:
        return 0, 0

    num_zeros = numpy.count_nonzero(weight == zero_point)

    return num_zeros, weight.size


def group_four_block(array: numpy.ndarray, pad_value: bool = True) -> numpy.ndarray:
    """
    :param array: array to group into four blocks
    :param pad_value: value to pad remainder block with
    :return: array grouped into blocks with shape [-1, 4]
    """
    # Transpose so input channels are last
    if array.ndim > 2:
        input_channel_dim = 1
    else:
        input_channel_dim = 0

    transpose_arg = list(range(array.ndim))
    del transpose_arg[input_channel_dim]
    transpose_arg.append(input_channel_dim)
    array = numpy.transpose(array, transpose_arg)

    # Pad array features with zeros to be divisible by four
    remainder = array.shape[-1] % 4
    if remainder != 0:
        num_missing_features = 4 - remainder
        array = numpy.pad(
            array,
            [(0, 0)] * (array.ndim - 1) + [(0, num_missing_features)],
            constant_values=True,
        )

    # Group into blocks and count zero blocks
    array_blocks = numpy.reshape(array, (-1, 4))
    return array_blocks


def get_node_num_four_block_zeros_and_size(
    model_graph: ONNXGraph, node: NodeProto
) -> Tuple[int, int]:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node whose four block sparsity sizes are being calculated
    :return: number of zero blocks and number of total blocks
    """
    # Get param and zero point
    zero_point = get_zero_point(model_graph, node)
    weight = get_node_weight(model_graph, node)
    if weight is None:
        return 0, 0

    # Group into blocks
    weight_blocks = group_four_block(weight, pad_value=zero_point)

    # Count non-zero blocks
    num_zeros_per_block = numpy.count_nonzero(weight_blocks == zero_point, axis=1)
    num_zero_blocks = numpy.count_nonzero(num_zeros_per_block == 4, axis=0)

    return num_zero_blocks, weight_blocks.shape[0]


def get_zero_point(model_graph: ONNXGraph, node: NodeProto) -> int:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node to find zero point of
    :return: zero point of given node
    """

    def _get_node_zero_point_init_name(node: NodeProto) -> str:
        if node.op_type in ["ConvInteger", "MatMulInteger"]:
            return _get_node_input(node, 3, default=None)
        if node.op_type == "QLinearConv":
            return _get_node_input(node, 5, default=None)
        if node.op_type == "QLinearMatMul":
            return _get_node_input(node, 7, default=None)
        raise Exception(
            "Node with op type {node.op_type} does not have a zero point initializer"
        )

    if node.op_type in ["Gather"]:
        return 0

    if is_quantized_layer(model_graph, node):
        zero_point_initializer_name = _get_node_zero_point_init_name(node)
        zero_point = get_initializer_value(
            model_graph, node, zero_point_initializer_name
        )
        if zero_point.ndim != 0:
            raise NotImplementedError("Channel-wise zero points are not supported")

        return int(zero_point)
    else:
        return 0


def is_sparse_layer(model_graph: ONNXGraph, node: NodeProto) -> bool:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node whose sparsity is being checked
    :return: true if node weights have any sparsity, False otherwise
    """
    return get_node_sparsity(model_graph, node) > 0


def is_four_block_sparse_layer(
    model_graph: ONNXGraph, node: NodeProto, threshold: int = 0.05
) -> bool:
    """
    Estimates if this node was likely pruned with four block sparsity.
    This is determined by comparing the difference between normal sparsity and
    four block sparsity and evaluating if the difference falls under a threshold

    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node whose four block sparsity is being checked
    :param threshold: threshold for measuring sparsity differences
    :return: true if node weights have any four block sparsity, False otherwise
    """
    four_block_sparsity = get_node_four_block_sparsity(model_graph, node)
    sparsity = get_node_sparsity(model_graph, node)
    return four_block_sparsity > 0 and abs(four_block_sparsity - sparsity) <= threshold


def is_quantized_layer(model_graph: ONNXGraph, node: NodeProto) -> bool:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node whose quantized status is being checked
    :return: True if node contains quantized weights, False otherwise
    """
    if node.op_type == "Gather":
        weight = get_node_weight(model_graph, node)
        return weight is not None and weight.dtype in [numpy.uint8, numpy.int8]

    return node.op_type in [
        "QLinearConv",
        "ConvInteger",
        "MatMulInteger",
        "QLinearMatMul",
    ]


def get_node_four_block_sparsity(model_graph: ONNXGraph, node: NodeProto) -> float:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node whose four block sparsity is being calculated
    :return: four block sparsity of node
    """

    num_zero_blocks, num_total_blocks = get_node_num_four_block_zeros_and_size(
        model_graph, node
    )
    if num_total_blocks == 0:
        return 0.0

    return float(num_zero_blocks / num_total_blocks)


def get_node_sparsity(model_graph: ONNXGraph, node: NodeProto) -> float:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node whose sparsity is being calculated
    :return: proportion of zeros in given node
    """
    num_zeros, weight_size = get_node_num_zeros_and_size(model_graph, node)
    if weight_size == 0:
        return 0.0

    # Embedding layer with one zero
    if num_zeros == 1 and node.op_type == "Gather":
        return 0.0

    return num_zeros / weight_size


def is_parameterized_prunable_layer(model_graph: ONNXGraph, node: NodeProto) -> bool:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node being checked
    :return: True if this node performs a operation that is parameterized and
        prunable, False otherwise
    """
    return get_node_weight(model_graph, node) is not None


def get_node_weight_name(model_graph: ONNXGraph, node: NodeProto) -> Union[str, None]:
    """
    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node that contains the weight
    :return: name of the weight initializer of the node, None if not found
    """
    initializer_names = model_graph.get_init_names()

    if node.op_type == "Gather":
        return _get_node_input(node, 0, default=None)

    if node.op_type in ["Conv", "ConvInteger"]:
        return _get_node_input(node, 1, default=None)

    if node.op_type == "QLinearConv":
        return _get_node_input(node, 3, default=None)

    if node.op_type == "QLinearMatMul":
        input_b_name = _get_node_input(node, 3, default=None)
        if input_b_name in initializer_names:
            return input_b_name

        input_a_name = _get_node_input(node, 0, default=None)
        if input_a_name in initializer_names:
            return input_a_name

    if node.op_type in ["MatMul", "Gemm", "MatMulInteger"]:
        return next(
            (
                input_name
                for input_name in node.input
                if input_name in initializer_names
            ),
            None,
        )

    return None


def get_node_weight(
    model_graph: ONNXGraph, node: NodeProto
) -> Union[numpy.ndarray, None]:
    """
    Finds parameter value of node (the node weight)

    :param model_graph: instance of ONNXGraph that contains the given node
    :param node: node to which parameter belongs to
    :return: a numpy array of param value, None if not found
    """

    initializer_name = get_node_weight_name(model_graph, node)
    weight = get_initializer_value(model_graph, node, initializer_name)
    if initializer_name is not None and weight is None and node.op_type != "Gather":
        raise Exception(f"Parameter for {node.name} not found")

    return weight


def get_layer_and_op_counts(
    model_graph: ONNXGraph,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Creates two dictionaries, each mapping op_type to the number of nodes of
    that op_type. The first dictionary contains op_types which are layers,
    the second contains op_types which are operations.

    :param model_graph: model graph whose counts are being checked
    :return: a layer dictionary and an operation dictionary which hold node counts
    """
    layer_counts = {}
    op_counts = {}

    for node in model_graph.nodes:
        target_dict = (
            layer_counts
            if is_parameterized_prunable_layer(model_graph, node)
            else op_counts
        )

        if node.op_type not in target_dict:
            target_dict[node.op_type] = 0

        target_dict[node.op_type] += 1

    return layer_counts, op_counts


def extract_node_id(node: NodeProto) -> str:
    """
    Get the node id for a given node from an ONNX model.
    Grabs the first ouput id as the node id.
    This is because is guaranteed to be unique for this node by the ONNX spec.

    :param node: the node to grab an id for
    :return: the id for the node
    """
    outputs = node.output

    return str(outputs[0])


def _get_node_input(
    node: NodeProto, index: int, default: Optional[Any] = None
) -> Union[str, Any]:
    """
    :param node: node that contains the desired input
    :param index: index of desired input
    :param default: default value if node.input does not contain index
    :return: the node input at the given index, default otherwise
    """
    if len(node.input) - 1 >= index:
        return node.input[index]
    else:
        return default
