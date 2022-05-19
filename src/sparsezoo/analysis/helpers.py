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

from typing import Union

import numpy as np
from onnx import ModelProto, NodeProto, numpy_helper


def get_initializer(model: ModelProto, initializer_name: str) -> np.ndarray:
    """
    Helper function to find initializers by name in the model graph
    :return: The initalizer if found, None otherwise
    """
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            return initializer

    return None


def get_zero_point(model: ModelProto, node: NodeProto) -> Union[int, np.ndarray]:
    """
    :TODO: Handle case where zero point is a tensor
    :return: The zero point of the node
    """
    if is_quantized_layer(node):
        initializer_name = node.input[3]
        initializer = get_initializer(model, initializer_name)
        return int(numpy_helper.to_array(initializer))
    else:
        return 0


def is_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if node weights have any sparsity, False otherwise
    """
    return get_node_sparsity(model, node) > 0


def is_four_block_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if node weights have any four block sparsity, False otherwise
    """
    return get_node_four_block_sparsity(model, node) > 0


def is_quantized_layer(node: NodeProto) -> bool:
    """
    :return: True if the node contains quantized weights, False otherwise
    """
    return node.op_type in ["ConvInteger", "MatMulInteger"]


def get_node_four_block_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :return: The four block sparsity of the node
    """

    # Get param and zero point
    zero_point = get_zero_point(model, node)
    param = get_layer_param(model, node)
    if param is None:
        return 0.0

    # Pad input channels with zeros
    num_missing_channels = param.shape[1] % 4
    padding_shape = [param.shape[0], num_missing_channels] + list(param.shape[2:])
    padding = np.zeros(padding_shape)
    param = np.concatenate((param, padding), axis=1)

    # TODO: Handle case if zero point is a tensor
    param_zeros = param == zero_point

    # Transpose so input channel is first
    transpose_arg = np.arange(param_zeros.ndim)
    transpose_arg[0] = 1
    transpose_arg[1] = 0
    param_zeros = np.transpose(param_zeros, transpose_arg)

    # Group parameter by blocks and count
    param_blocks = np.resize(param_zeros, (-1, 4))
    num_zeros_per_block = np.count_nonzero(param_blocks, axis=1)
    num_zero_blocks = np.count_nonzero(num_zeros_per_block, axis=0)

    return num_zero_blocks / param_blocks.shape[0]


def get_node_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :return: The proportion of zeros in the given node
    """
    zero_point = get_zero_point(model, node)
    param = get_layer_param(model, node)
    if param is None:
        return 0.0

    num_zeros = np.count_nonzero(param == zero_point)

    return num_zeros / param.size


def is_parameterized_prunable_layer(model: ModelProto, node: NodeProto) -> bool:
    """
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
    Finds the parameter value of the node
    :return: A numpy array of the param value, None if not found
    """

    if node.op_type == "Conv":
        initializer_name = node.input[1]
        initializer = get_initializer(model, initializer_name)
        if initializer is None:
            raise Exception("Parameter not found")

        return numpy_helper.to_array(initializer)

    elif node.op_type in ["MatMul", "Gemm", "MatMulInteger"]:
        initializer_names = [init.name for init in model.graph.initializer]
        initializer_name = next(
            name for name in node.input if name in initializer_names
        )

        initializer = get_initializer(model, initializer_name)
        if initializer is None:
            raise Exception("Parameter not found")

        return numpy_helper.to_array(initializer)

    else:
        return None


def get_layer_and_op_counts(model: ModelProto):
    """
    Creates two dictionaries, each mapping op_type to the number of nodes of
        that op_type. The first dictionary contains op_types which are layers,
        the second contains op_types which are operations.
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
