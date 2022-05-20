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

import numpy as np
from onnx import ModelProto, NodeProto, numpy_helper


def get_initializer(model: ModelProto, initializer_name: str) -> np.ndarray:
    """
    Helper function to find initializers by name in the model graph

    :param model: The model used to find the initializer
    :param initializer_name: The name of the initializer being returned
    :return: The initalizer if found, None otherwise
    """
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            return initializer

    return None


def get_node_sparsity_sizes(model: ModelProto, node: NodeProto) -> Tuple[int, int]:
    """
    :param model: The model in which the node's parameter exists
    :param node: The node whose number of zeros and parameter size is being calculated
    :return: The number of zeros, the number of total values
    """
    zero_point = get_zero_point(model, node)
    param = get_layer_param(model, node)
    if param is None:
        return 0, 0

    num_zeros = np.count_nonzero(param == zero_point)

    return num_zeros, param.size


def get_node_four_block_sparsity_sizes(
    model: ModelProto, node: NodeProto
) -> Tuple[int, int]:
    """
    :param model: The model in which the node's parameter exists
    :param node: The node whose four block sparsity sizes are being calculated
    :return: The number of zero blocks, the number of total blocks
    """
    # Get param and zero point
    zero_point = get_zero_point(model, node)
    param = get_layer_param(model, node)
    if param is None:
        return 0, 0

    # Bool array
    param_zeros = param == zero_point

    # Transpose so input channels are first
    transpose_arg = np.arange(param_zeros.ndim)
    transpose_arg[0] = 1
    transpose_arg[1] = 0
    param_zeros = np.transpose(param_zeros, transpose_arg)

    # Flatten and treat weights as features across input channels
    param_zeros = np.reshape(param_zeros, (param_zeros.shape[0], -1))

    # Pad weight features with zeros to be divisible by four
    num_missing_features = (4 - (param_zeros.shape[1] % 4)) % 4
    if num_missing_features != 0:
        param_zeros = np.pad(
            param_zeros, ((0, 0), (0, num_missing_features)), constant_values=True
        )

    # Group into blocks and count full blocks
    param_blocks = np.reshape(param_zeros, (-1, 4))
    num_zeros_per_block = np.count_nonzero(param_blocks, axis=1)
    num_zero_blocks = np.count_nonzero(num_zeros_per_block == 4, axis=0)

    return num_zero_blocks, param_blocks.shape[0]


def get_zero_point(model: ModelProto, node: NodeProto) -> int:
    """
    :param model: The model in which the node's zero point initializer exists
    :param node: The node to find the zero point of
    :return: The zero point of the given node
    """
    if is_quantized_layer(node):
        initializer_name = node.input[3]
        initializer = get_initializer(model, initializer_name)
        zero_point_initializer = numpy_helper.to_array(initializer)
        if zero_point_initializer.ndim != 0:
            raise NotImplementedError("Layer-wise zero points are not supported")

        return int(zero_point_initializer)
    else:
        return 0


def is_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :param model: The model in which the node's parameter exists
    :param node: The node whose sparsity is being checked
    :return: True if node weights have any sparsity, False otherwise
    """
    return get_node_sparsity(model, node) > 0


def is_four_block_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :param model: The model in which the node's parameter exists
    :param node: The node whose four block sparsity is being checked
    :return: True if node weights have any four block sparsity, False otherwise
    """
    return get_node_four_block_sparsity(model, node) > 0


def is_quantized_layer(node: NodeProto) -> bool:
    """
    :param node: The node whose quantized status is being checked
    :return: True if the node contains quantized weights, False otherwise
    """
    return node.op_type in ["QLinearConv", "ConvInteger", "MatMulInteger"]


def get_node_four_block_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :param model: The model in which the node's parameter exists
    :param node: The node whose four block sparsity is being calculated
    :return: The four block sparsity of the node
    """

    num_zero_blocks, num_total_blocks = get_node_four_block_sparsity_sizes(model, node)
    if num_total_blocks == 0:
        return 0.0

    return num_zero_blocks / num_total_blocks


def get_node_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :param model: The model in which the node's parameter exists
    :param node: The node whose sparsity is being calculated
    :return: The proportion of zeros in the given node
    """
    num_zeros, param_size = get_node_sparsity_sizes(model, node)
    if param_size == 0:
        return 0.0

    return num_zeros / param_size


def is_parameterized_prunable_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :param model: The model in which the node's parameter exists
    :param node: The node being checked
    :return: True if this node performs a operation that is parameterized and
        prunable, False otherwise
    """
    return (
        node.op_type
        in [
            "Conv",
            "MatMul",
            "Gemm",
            "QLinearConv",
            "QLinearMatMul",
            "ConvInteger",
            "MatMulInteger",
            "Gather",
        ]
        and not get_layer_param(model, node) is None
    )


def get_layer_param(model: ModelProto, node: NodeProto) -> np.ndarray:
    """
    Finds the parameter value of the node (the node's weight)

    :param model: The model in which the node's parameter exists
    :param node: The node to which the parameter belongs to
    :return: A numpy array of the param value, None if not found
    """

    if node.op_type in ["Conv", "ConvInteger"]:
        initializer_name = node.input[1]

    elif node.op_type in ["QLinearConv"]:
        initializer_name = node.input[3]

    elif node.op_type in ["MatMul", "Gemm", "MatMulInteger"]:
        initializer_names = [init.name for init in model.graph.initializer]
        initializer_name = next(
            name for name in node.input if name in initializer_names
        )

    else:
        return None

    initializer = get_initializer(model, initializer_name)
    if initializer is None:
        raise Exception("Parameter not found")

    return numpy_helper.to_array(initializer)


def get_layer_and_op_counts(model: ModelProto):
    """
    Creates two dictionaries, each mapping op_type to the number of nodes of
        that op_type. The first dictionary contains op_types which are layers,
        the second contains op_types which are operations.

    :param model: The model whose counts are being checked
    :return: a layer dictionary and an operation dictionary which hold node counts
    """
    model_op_types = [node.op_type for node in model.graph.node]

    layer_counts = {}
    op_counts = {}
    for op_type in model_op_types:
        op_type_nodes = [node for node in model.graph.node if node.op_type == op_type]
        op_count = len(op_type_nodes)
        assert len(op_type_nodes) > 0

        if is_parameterized_prunable_layer(model, op_type_nodes[0]):
            layer_counts[op_type] = op_count

        else:
            op_counts[op_type] = op_count

    return layer_counts, op_counts
