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

from typing import Tuple

import numpy
from onnx import ModelProto, NodeProto, numpy_helper

from sparsezoo.analysis.onnx_helpers import (
    calculate_num_operations,
    extract_node_id,
    extract_node_shapes,
    get_kernel_shape,
    get_node_attributes,
)


__all__ = [
    "get_layer_and_op_counts",
    "get_node_four_block_sparsity",
    "get_node_num_four_block_zeros_and_size",
    "get_node_sparsity",
    "get_node_weight",
    "get_node_bias",
    "get_node_num_zeros_and_size",
    "get_zero_point",
    "is_four_block_sparse_layer",
    "is_parameterized_prunable_layer",
    "is_quantized_layer",
    "is_sparse_layer",
    "get_num_operations",
]

def _get_node_input(node, index, default=None):
    """
    :param node: node that contains the desired input
    :param index: index of desired input
    :param default: default value if node.input does not contain index
    """
    if len(node.input) -1 >= index:
        return node.input[index]
    else:
        return default


def get_node_bias(model: ModelProto, node: NodeProto) -> numpy.ndarray:
    """
    Finds parameter value of node (the node weight)

    :param model: model that contains the given node
    :param node: node to which parameter belongs to
    :return: a numpy array of param value, None if not found
    """

    def _get_node_bias_name(model: ModelProto, node: NodeProto) -> str:
        initializer_names = [init.name for init in model.graph.initializer]

        if node.op_type in ["Conv", "Gemm"]:
            return _get_node_input(node, 2, default=None)

        if node.op_type == "QLinearConv":
            return _get_node_input(node, 8, default=None)

        return None

    initializer_name = _get_node_bias_name(model, node)
    if initializer_name is None:
        return None
    initializer = get_initializer(model, initializer_name)
    if initializer is None:
        raise Exception(f"Parameter for {node.name} not found")

    return numpy_helper.to_array(initializer)


def get_num_operations(model: ModelProto, node: NodeProto) -> int:
    """
    Gets an approximation of the number of floating point or integer operations

    :param model: model that contains the given node
    :param node: node which performs the operations
    :return: number of operations performed by node
    """

    node_shapes = extract_node_shapes(model)
    attributes = get_node_attributes(node)

    weight = get_node_weight(model, node)
    bias = get_node_bias(model, node)
    weight_shape = list(weight.shape) if weight is not None else None
    bias_shape = list(bias.shape) if bias is not None else None

    node_shape = node_shapes.get(extract_node_id(node))
    input_shapes = node_shape.input_shapes if node_shape is not None else None
    output_shapes = node_shape.output_shapes if node_shape is not None else None

    kernel_shape = get_kernel_shape(attributes)

    num_operations = calculate_num_operations(
        node.op_type,
        input_shape=input_shapes,
        output_shape=output_shapes,
        weight_shape=weight_shape,
        kernel_shape=kernel_shape,
        bias_shape=bias_shape,
        attributes=attributes,
    )

    return int(num_operations) if num_operations is not None else 0


def get_initializer(model: ModelProto, initializer_name: str) -> numpy.ndarray:
    """
    Helper function to find initializers by name in model graph

    :param model: model that contains the initializer with the given name
    :param initializer_name: name of initializer being returned
    :return: initalizer if found, None otherwise
    """
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            return initializer

    return None


def get_node_num_zeros_and_size(model: ModelProto, node: NodeProto) -> Tuple[int, int]:
    """
    :param model: model that contains the given node
    :param node: node whose number of zeros and parameter size is being calculated
    :return: number of zeros and number of total values in node parameter
    """
    zero_point = get_zero_point(model, node)
    weight = get_node_weight(model, node)
    if weight is None:
        return 0, 0

    num_zeros = numpy.count_nonzero(weight == zero_point)

    return num_zeros, weight.size


def get_node_num_four_block_zeros_and_size(
    model: ModelProto, node: NodeProto
) -> Tuple[int, int]:
    """
    :param model: model that contains the given node
    :param node: node whose four block sparsity sizes are being calculated
    :return: number of zero blocks and number of total blocks
    """
    # Get param and zero point
    zero_point = get_zero_point(model, node)
    weight = get_node_weight(model, node)
    if weight is None:
        return 0, 0

    # Bool array
    weight_zeros = weight == zero_point

    # Transpose so input channels are last
    if weight_zeros.ndim > 2 or (node.op_type == "Gemm" and "transB" in node.attribute):
        input_channel_dim = 1
    else:
        input_channel_dim = 0

    transpose_arg = list(range(weight_zeros.ndim))
    del transpose_arg[input_channel_dim]
    transpose_arg.append(input_channel_dim)
    weight_zeros = numpy.transpose(weight_zeros, transpose_arg)

    # Pad weight features with zeros to be divisible by four
    remainder = weight_zeros.shape[-1] % 4
    if remainder != 0:
        num_missing_features = 4 - remainder
        weight_zeros = numpy.pad(
            weight_zeros,
            [(0, 0)] * (weight_zeros.ndim - 1) + [(0, num_missing_features)],
            constant_values=True,
        )

    # Group into blocks and count zero blocks
    weight_blocks = numpy.reshape(weight_zeros, (-1, 4))
    num_zeros_per_block = numpy.count_nonzero(weight_blocks, axis=1)
    num_zero_blocks = numpy.count_nonzero(num_zeros_per_block == 4, axis=0)

    return num_zero_blocks, weight_blocks.shape[0]


def get_zero_point(model: ModelProto, node: NodeProto) -> int:
    """
    :param model: model that contains the given node
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
            "Node with op type {node.op_type} does not have a zero " "point initializer"
        )

    if node.op_type in ["Gather"]:
        return 0

    if is_quantized_layer(model, node):
        zero_point_initializer_name = _get_node_zero_point_init_name(node)
        zero_point_initializer = get_initializer(model, zero_point_initializer_name)
        zero_point = numpy_helper.to_array(zero_point_initializer)
        if zero_point.ndim != 0:
            raise NotImplementedError("Channel-wise zero points are not supported")

        return int(zero_point)
    else:
        return 0


def is_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :param model: model that contains the given node
    :param node: node whose sparsity is being checked
    :return: true if node weights have any sparsity, False otherwise
    """
    return get_node_sparsity(model, node) > 0


def is_four_block_sparse_layer(
    model: ModelProto, node: NodeProto, threshold: int = 0.05
) -> bool:
    """
    :param model: model that contains the given node
    :param node: node whose four block sparsity is being checked
    :return: true if node weights have any four block sparsity, False otherwise
    """
    four_block_sparsity = get_node_four_block_sparsity(model, node)
    sparsity = get_node_sparsity(model, node)
    return four_block_sparsity > 0 and abs(four_block_sparsity - sparsity) <= threshold


def is_quantized_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :param node: node whose quantized status is being checked
    :return: true if node contains quantized weights, False otherwise
    """
    if node.op_type == "Gather":
        weight = get_node_weight(model, node)
        return weight is not None and weight.dtype in [numpy.uint8, numpy.int8]

    return node.op_type in [
        "QLinearConv",
        "ConvInteger",
        "MatMulInteger",
        "QLinearMatMul",
    ]


def get_node_four_block_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :param model: model that contains the given node
    :param node: node whose four block sparsity is being calculated
    :return: four block sparsity of node
    """

    num_zero_blocks, num_total_blocks = get_node_num_four_block_zeros_and_size(
        model, node
    )
    if num_total_blocks == 0:
        return 0.0

    return num_zero_blocks / num_total_blocks


def get_node_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :param model: model that contains the given node
    :param node: node whose sparsity is being calculated
    :return: proportion of zeros in given node
    """
    num_zeros, weight_size = get_node_num_zeros_and_size(model, node)
    if weight_size == 0:
        return 0.0

    return num_zeros / weight_size


def is_parameterized_prunable_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :param model: model that contains the given node
    :param node: node being checked
    :return: True if this node performs a operation that is parameterized and
        prunable, False otherwise
    """
    return get_node_weight(model, node) is not None


def get_node_weight(model: ModelProto, node: NodeProto) -> numpy.ndarray:
    """
    Finds parameter value of node (the node weight)

    :param model: model that contains the given node
    :param node: node to which parameter belongs to
    :return: a numpy array of param value, None if not found
    """

    def _get_node_weight_name(model: ModelProto, node: NodeProto) -> str:
        initializer_names = [init.name for init in model.graph.initializer]

        if node.op_type == "Gather":
            return _get_node_input(node, 0, default=None)

        if node.op_type in ["Conv", "ConvInteger"]:
            return _get_node_input(node, 1, default=None)

        if node.op_type == "QLinearConv":
            return _get_node_input(node, 3, default=None)

        if node.op_type == "QLinearMatMul":
            if _get_node_input(node, 3, default=None) in initializer_names:
                return _get_node_input(node, 3, default=None)

            if _get_node_input(node, 0, default=None) in initializer_names:
                return _get_node_input(node, 0, default=None)

        if node.op_type in ["MatMul", "Gemm", "MatMulInteger"]:
            return next(
                input_name
                for input_name in node.input
                if input_name in initializer_names
            )

        return None

    initializer_name = _get_node_weight_name(model, node)
    if initializer_name is None:
        return None
    initializer = get_initializer(model, initializer_name)
    if initializer is None:
        if node.op_type in ["Gather"]:
            return None
        else:
            raise Exception(f"Parameter for {node.name} not found")

    return numpy_helper.to_array(initializer)


def get_layer_and_op_counts(model: ModelProto):
    """
    Creates two dictionaries, each mapping op_type to the number of nodes of
        that op_type. The first dictionary contains op_types which are layers,
        the second contains op_types which are operations.

    :param model: model whose counts are being checked
    :return: a layer dictionary and an operation dictionary which hold node counts
    """
    layer_counts = {}
    op_counts = {}

    for node in model.graph.node:
        target_dict = (
            layer_counts if is_parameterized_prunable_layer(model, node) else op_counts
        )

        if node.op_type not in target_dict:
            target_dict[node.op_type] = 0

        target_dict[node.op_type] += 1

    return layer_counts, op_counts
