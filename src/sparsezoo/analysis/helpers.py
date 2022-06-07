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

from typing import Any, Dict, List, Optional, Tuple

import numpy
from onnx import ModelProto, NodeProto, numpy_helper
from onnx.helper import get_attribute_value

from sparsezoo.analysis.node_shape import (
    NodeShape,
    extract_node_id,
    extract_node_shapes,
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
    "get_num_dense_and_sparse_ops",
    "NodeShape",
    "extract_node_shapes",
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


def get_node_bias(model: ModelProto, node: NodeProto) -> numpy.ndarray:
    """
    Finds parameter value of node (the node weight)

    :param model: model that contains the given node
    :param node: node to which parameter belongs to
    :return: a numpy array of param value, None if not found
    """

    def _get_node_bias_name(model: ModelProto, node: NodeProto) -> str:
        if node.op_type in ["Conv", "Gemm"]:
            return _get_node_input(node, 2, default=None)

        if node.op_type == "QLinearConv":
            return _get_node_input(node, 8, default=None)

        return None

    initializer_name = _get_node_bias_name(model, node)
    if initializer_name is None:
        return None

    return get_initializer_value(model, node, initializer_name)


def get_num_dense_and_sparse_ops(
    model: ModelProto,
    node: NodeProto,
    node_shapes: Optional[Dict[str, NodeShape]] = None,
    is_four_block_sparse: Optional[bool] = None,
) -> Tuple[int, int]:
    """
    Gets an approximation of the number of floating point or integer operations

    :param model: model that contains the given node
    :param node: node which performs the operations
    :return: number of operations performed by node
    """
    if node_shapes is None:
        node_shapes = extract_node_shapes(model)

    node_shape = node_shapes.get(extract_node_id(node))
    input_shapes = node_shape.input_shapes if node_shape is not None else None
    output_shapes = node_shape.output_shapes if node_shape is not None else None

    weight = get_node_weight(model, node)
    bias = get_node_bias(model, node)
    zero_point = get_zero_point(model, node)
    if is_four_block_sparse is None:
        is_four_block_sparse = is_four_block_sparse_layer(model, node)

    node_attributes = get_node_attributes(node)

    if node.op_type in ["Add", "Mul", "Div", "Sub", "Clip"] or node.op_type in [
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Tanh",
    ]:
        return (numpy.prod(output_shapes), 0) if output_shapes is not None else (0, 0)

    # If BN is followed by matmul or conv, then it is folded into the following
    # layer weights. Assume this is true for all cases
    if node.op_type == "BatchNormalization":
        return (0, 0)

    if node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
        return (numpy.prod(input_shapes), 0) if input_shapes is not None else (0, 0)

    if node.op_type in ["MaxPool", "AveragePool"]:
        if "kernel_shape" not in node_attributes:
            raise Exception(
                f"No kernel_shape found in node attributes of {node.op_type}"
            )
        kernel_shape = node_attributes["kernel_shape"]
        return (
            (numpy.prod(output_shapes) * numpy.prod(kernel_shape), 0)
            if output_shapes is not None
            else None
        )

    if node.op_type in ["Gemm", "MatMul", "MatMulInteger", "QLinearMatMul"]:
        if input_shapes is None or len(input_shapes) == 0:
            return (0, 0)
        input_shape = input_shapes[0]

        # If no weight supplied, treat other input as dense weight
        # TODO: Paste asana talk. When runtime implements activation sparsity
        if weight is None:
            weight_shape = input_shapes[1]
            weight = numpy.full(weight_shape, zero_point - 1)

        # Weight operations
        num_dense_ops, num_sparse_ops = _get_gemm_dense_sparse_ops(
            weight,
            input_shape,
            zero_point=zero_point,
            is_four_block_sparse=is_four_block_sparse,
        )

        # Bias operations
        if bias is not None:
            bias_dense_ops, bias_sparse_ops = _get_bias_dense_sparse_ops(output_shapes)
            num_dense_ops += bias_dense_ops
            num_sparse_ops += bias_sparse_ops

        return num_dense_ops, num_sparse_ops

    if node.op_type in ["Conv", "ConvInteger", "QLinearConv"]:
        input_shape = input_shapes[0]
        output_shape = output_shapes[0]
        pads = node_attributes["pads"] if "pads" in node_attributes else [0, 0, 0, 0]
        strides = node_attributes["strides"] if "strides" in node_attributes else [1, 1]
        group = node_attributes["group"] if "group" in node_attributes else 1

        # Weight operations
        num_dense_ops, num_sparse_ops = _get_conv_weight_dense_sparse_ops(
            weight,
            input_shape,
            pads,
            strides,
            group,
            zero_point=zero_point,
            is_four_block_sparse=is_four_block_sparse,
        )

        # Bias operations
        if bias is not None:
            bias_dense_ops, bias_sparse_ops = _get_bias_dense_sparse_ops(output_shapes)
            num_dense_ops += bias_dense_ops
            num_sparse_ops += bias_sparse_ops

        return num_dense_ops, num_sparse_ops

    return 0, 0


def get_initializer_value(
    model: ModelProto, node: NodeProto, initializer_name: str
) -> numpy.ndarray:
    """
    Helper function to find initializers by name in model graph

    :param model: model that contains the initializer with the given name
    :param initializer_name: name of initializer being returned
    :return: initalizer if found, None otherwise
    """

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

    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            break
    else:
        return None

    value = numpy_helper.to_array(initializer)
    if _is_transposed_initializer(node, initializer_name):
        value = value.T

    return value


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


def group_four_block(array, pad_value=True):
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

    # Group into blocks
    weight_blocks = group_four_block(weight, pad_value=zero_point)

    # Count non-zero blocks
    num_zeros_per_block = numpy.count_nonzero(weight_blocks == zero_point, axis=1)
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
        zero_point = get_initializer_value(model, node, zero_point_initializer_name)
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

    # Embedding layer with one zero
    if num_zeros == 1 and node.op_type == "Gather":
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
            input_b_name = _get_node_input(node, 3, default=None)
            if input_b_name in initializer_names:
                return input_b_name

            input_a_name = _get_node_input(node, 0, default=None)
            if input_a_name in initializer_names:
                return input_a_name

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

    weight = get_initializer_value(model, node, initializer_name)
    if weight is None and node.op_type != "Gather":
        raise Exception(f"Parameter for {node.name} not found")

    return weight


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


def _get_node_input(node: NodeProto, index: int, default: Optional[Any] = None):
    """
    :param node: node that contains the desired input
    :param index: index of desired input
    :param default: default value if node.input does not contain index
    """
    if len(node.input) - 1 >= index:
        return node.input[index]
    else:
        return default


def _get_gemm_dense_sparse_ops(
    weight: numpy.ndarray,
    input_shape: List[int],
    zero_point: int = 0,
    is_four_block_sparse: bool = False,
) -> Tuple[int, int]:
    """
    Calculates number of operations performed by performing matrix multiplication

    :param weight: input matrix to matmul (with parameterized weights)
    :param input_shape: shape of other input matrix
    :param zero_point: number representing zero value of weight
    :param is_four_block_sparse: true if the weight is four block sparse
    :return: number of dense and sparse operations performed
    """
    if is_four_block_sparse:
        weight_blocks = group_four_block(weight, pad_value=zero_point)
        num_zeros_per_block = numpy.count_nonzero(weight_blocks == zero_point, axis=1)

        num_zero_blocks = numpy.count_nonzero(num_zeros_per_block == 4, axis=0)
        num_non_zero_blocks = numpy.count_nonzero(num_zeros_per_block != 4, axis=0)

        num_dense_ops = 2 * input_shape[-2] * num_non_zero_blocks * 4
        num_sparse_ops = 2 * input_shape[-2] * num_zero_blocks * 4
    else:
        num_zeros = numpy.count_nonzero(weight == zero_point)
        num_non_zeros = numpy.count_nonzero(weight != zero_point)

        num_dense_ops = 2 * input_shape[-2] * num_non_zeros
        num_sparse_ops = 2 * input_shape[-2] * num_zeros

    return num_dense_ops, num_sparse_ops


def _get_kernel_subview(
    weight: numpy.ndarray,
    x: int,
    y: int,
    spatial_shape: List[int],
    kernel_shape: List[int],
    pads: List[int],
) -> Tuple[int, int, int, int]:
    """
    Calculates which spatial coordinates in the kernel will be applied to a pixel
    at coordinates (x, y) and returns the coordinates of a subview of the kernel
    containing only those spatial coordinates

    :param weight: matrix with shape [out_channels, in_channels, kernel_h, kernel_w]
    :param x: x coordinate of pixel in the input
    :param y: y coordinate of pixel in the input
    :param spatial_shape: spatial dimensions of input
    :param kernel_shape: spatial dimensions of kernel
    :param pads: list of paddings around the input
    :return: the coordinates of a subview of the kernel containing only the
    coordinates that will be multiplied with the pixel at coordinate (x, y)
    """
    distance_from_bottom = spatial_shape[0] - y - 1
    y0 = kernel_shape[0] - pads[3] - distance_from_bottom - 1
    y0 = max(y0, 0)
    y1 = kernel_shape[0] + pads[1] + y
    y1 = min(y1, kernel_shape[0])

    distance_from_right = spatial_shape[1] - x - 1
    x0 = kernel_shape[1] - pads[2] - distance_from_right - 1
    x0 = max(x0, 0)
    x1 = kernel_shape[1] + pads[0] + x
    x1 = min(x1, kernel_shape[1])

    return y0, y1, x0, x1


def _get_conv_weight_dense_sparse_ops(
    weight: numpy.ndarray,
    input_shape: List[int],
    pads: List[int],
    strides: List[int],
    group: int,
    zero_point: int = 0,
    is_four_block_sparse: bool = False,
) -> Tuple[int, int]:
    """
    Calculates number of operations performed by applying a convolutional weight

    :param weight: matrix of values to be convoluted
    :param input_shape: dimensions of the input to the weight operation
    :param pads: padding for spatial axes, [L, T, R, B]
    :param strides: stride along spatial axes
    :param group: number of groups input channels and output channels are divided into
    :param zero_point: number representing zero value of weight
    :param is_four_block_sparse: true if the weight is four block sparse
    :return: number of dense and sparse operations performed
    """
    spatial_shape = input_shape[2:]
    kernel_shape = weight.shape[2:]

    weight_spatial_flattened = numpy.reshape(weight, (*weight.shape[:2], -1))
    kernel_dense_sparse_ops = [
        _get_gemm_dense_sparse_ops(
            weight_spatial_flattened[:, :, i],
            [1, weight.shape[1]],
            zero_point,
            is_four_block_sparse,
        )
        for i in range(weight_spatial_flattened.shape[2])
    ]
    kernel_dense_sparse_ops = numpy.reshape(kernel_dense_sparse_ops, (*kernel_shape, 2))

    dense_sum, sparse_sum = 0, 0

    # TODO: This can be sped up by augmenting coordinates according to padding
    # For each pixel in the input
    for x in range(0, spatial_shape[1], strides[1]):
        for y in range(0, spatial_shape[0], strides[0]):

            # Calculate a subview of the kernels which contains only the
            # coordinates which apply to this pixel
            kernel_subview_coords = _get_kernel_subview(
                weight, x, y, spatial_shape, kernel_shape, pads
            )

            # Get the number of dense and sparse ops associated with each pixel
            # in the kernel subview
            kernel_ops_subview = kernel_dense_sparse_ops[
                kernel_subview_coords[0] : kernel_subview_coords[1],
                kernel_subview_coords[2] : kernel_subview_coords[3],
            ]

            dense_sum += numpy.sum(kernel_ops_subview[:, :, 0])
            sparse_sum += numpy.sum(kernel_ops_subview[:, :, 1])

    # Adjust for depthwise convolutions
    dense_sum = dense_sum // group
    sparse_sum = sparse_sum // group

    return dense_sum, sparse_sum


def _get_bias_dense_sparse_ops(output_shapes: List[List[int]]) -> Tuple[int, int]:
    """
    Calculates number of operations performed by applying a bias

    :param output_shapes: Shape of output of bias step
    :return:
    """
    return numpy.prod(output_shapes), 0
