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

import torch

import numpy
from onnx import ModelProto, NodeProto, numpy_helper


__all__ = [
    "get_layer_and_op_counts",
    "get_node_four_block_sparsity",
    "get_node_num_four_block_zeros_and_size",
    "get_node_sparsity",
    "get_layer_param",
    "get_node_num_zeros_and_size",
    "get_zero_point",
    "is_four_block_sparse_layer",
    "is_parameterized_prunable_layer",
    "is_quantized_layer",
    "is_sparse_layer",
]


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
    param = get_layer_param(model, node)
    if param is None:
        return 0, 0

    num_zeros = numpy.count_nonzero(param == zero_point)

    return num_zeros, param.size


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
    param = get_layer_param(model, node)
    if param is None:
        return 0, 0

    # Bool array
    param_zeros = param == zero_point

    # Transpose so input channels are last
    if param_zeros.ndim > 2 or (node.op_type == "Gemm" and "transB" in node.attribute):
        input_channel_dim = 1
    else:
        input_channel_dim = 0

    transpose_arg = list(range(param_zeros.ndim))
    del transpose_arg[input_channel_dim]
    transpose_arg.append(input_channel_dim)
    param_zeros = numpy.transpose(param_zeros, transpose_arg)

    # Pad weight features with zeros to be divisible by four
    remainder = param_zeros.shape[-1] % 4
    if remainder != 0:
        num_missing_features = 4 - remainder
        param_zeros = numpy.pad(
            param_zeros, [(0, 0)] * (param_zeros.ndim -1) + [(0, num_missing_features)], constant_values=True
        )

    # Group into blocks and count zero blocks
    param_blocks = numpy.reshape(param_zeros, (-1, 4))
    num_zeros_per_block = numpy.count_nonzero(param_blocks, axis=1)
    num_zero_blocks = numpy.count_nonzero(num_zeros_per_block == 4, axis=0)

    return num_zero_blocks, param_blocks.shape[0]


def get_zero_point(model: ModelProto, node: NodeProto) -> int:
    """
    :param model: model that contains the given node
    :param node: node to find zero point of
    :return: zero point of given node
    """
    def _get_node_zero_point_init_name(node: NodeProto) -> str:
        if node.op_type in ["ConvInteger", "MatMulInteger"]:
            return node.input[3]
        if node.op_type == "QLinearConv":
            return node.input[5]
        if node.op_type == "QLinearMatMul":
            return node.input[7]
        raise Exception("Node with op type {node.op_type} does not have a zero "
                        "point initializer")

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


def is_four_block_sparse_layer(model: ModelProto, node: NodeProto, threshold: int = 0.05) -> bool:
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
        param = get_layer_param(model, node)
        return param is not None and param.dtype in [numpy.uint8, numpy.int8]

    return node.op_type in ["QLinearConv", "ConvInteger", "MatMulInteger", "QLinearMatMul"]


def get_node_four_block_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :param model: model that contains the given node
    :param node: node whose four block sparsity is being calculated
    :return: four block sparsity of node
    """

    num_zero_blocks, num_total_blocks = get_node_num_four_block_zeros_and_size(model, node)
    if num_total_blocks == 0:
        return 0.0

    return num_zero_blocks / num_total_blocks


def get_node_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :param model: model that contains the given node
    :param node: node whose sparsity is being calculated
    :return: proportion of zeros in given node
    """
    num_zeros, param_size = get_node_num_zeros_and_size(model, node)
    if param_size == 0:
        return 0.0

    return num_zeros / param_size


def is_parameterized_prunable_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :param model: model that contains the given node
    :param node: node being checked
    :return: True if this node performs a operation that is parameterized and
        prunable, False otherwise
    """
    return get_layer_param(model, node) is not None


def get_layer_param(model: ModelProto, node: NodeProto) -> numpy.ndarray:
    """
    Finds parameter value of node (the node weight)

    :param model: model that contains the given node
    :param node: node to which parameter belongs to
    :return: a numpy array of param value, None if not found
    """
    def _get_layer_param_name(model: ModelProto, node: NodeProto) -> str:
        initializer_names = [init.name for init in model.graph.initializer]

        if node.op_type == "Gather":
            return node.input[0]

        if node.op_type in ["Conv", "ConvInteger"]:
            return node.input[1]

        if node.op_type == "QLinearConv":
            return node.input[3]

        if node.op_type == "QLinearMatMul":
            if node.input[3] in initializer_names:
                return node.input[3]

            if node.input[0] in initializer_names:
                return node.input[0]

        if node.op_type in ["MatMul", "Gemm", "MatMulInteger"]:
            return next(
                input_name for input_name in node.input if input_name in initializer_names
            )

        return None

    initializer_name = _get_layer_param_name(model, node)
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
    model_op_types = [node.op_type for node in model.graph.node]

    layer_counts = {}
    op_counts = {}

    for node in model.graph.node:
        target_dict = layer_counts if is_parameterized_prunable_layer(model, node) else op_counts

        if not node.op_type in target_dict:
            target_dict[node.op_type] = 0

        target_dict[node.op_type] += 1

    return layer_counts, op_counts
