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

from typing import Dict, List, Optional, Tuple, Union

import numpy
from onnx import ModelProto, NodeProto

from sparsezoo.analysis.utils.helpers import (
    extract_node_id,
    get_node_attributes,
    get_node_bias,
    get_node_weight,
    get_zero_point,
    group_four_block,
    is_four_block_sparse_layer,
)
from sparsezoo.analysis.utils.node_shape import NodeShape, extract_node_shapes


__all__ = [
    "get_num_dense_and_sparse_ops",
]


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
    :param node_shapes: optional dictionary of node shapes. If not supplied,
        node_shapes will be computed
    :param is_four_block_sparse: optional boolean indicating if this node is four
        block sparse. If not supplied, it be will be computed
    :return: number of operations performed by node
    """
    node_shapes = extract_node_shapes(model) if node_shapes is None else node_shapes

    node_shape = node_shapes.get(extract_node_id(node))
    input_shapes = node_shape.input_shapes if node_shape is not None else None
    output_shapes = node_shape.output_shapes if node_shape is not None else None

    weight = get_node_weight(model, node)
    bias = get_node_bias(model, node)
    zero_point = get_zero_point(model, node)
    is_four_block_sparse = (
        is_four_block_sparse_layer(model, node)
        if is_four_block_sparse is None
        else is_four_block_sparse
    )

    node_attributes = get_node_attributes(node)

    if node.op_type in [
        "Add",
        "Mul",
        "Div",
        "Sub",
        "Clip",
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Tanh",
    ]:
        return (_numpy_prod_none_safe(output_shapes), 0)

    # If BN is followed by matmul or conv, then it is folded into the following
    # layer weights. Assume this is true for all cases
    if node.op_type == "BatchNormalization":
        return (0, 0)

    if node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
        return (_numpy_prod_none_safe(input_shapes), 0)

    if node.op_type in ["MaxPool", "AveragePool"]:
        if "kernel_shape" not in node_attributes:
            raise Exception(
                f"No kernel_shape found in node attributes of {node.op_type}"
            )
        kernel_shape = node_attributes["kernel_shape"]
        return (
            _numpy_prod_none_safe(output_shapes) * _numpy_prod_none_safe(kernel_shape),
            0,
        )

    if node.op_type in ["Gemm", "MatMul", "MatMulInteger", "QLinearMatMul"]:
        if input_shapes is None or len(input_shapes) == 0:
            return (0, 0)
        input_shape = input_shapes[0]

        # If no weight supplied, treat other input as dense weight
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
        if input_shapes is None or len(input_shapes) == 0:
            return (0, 0)
        input_shape = input_shapes[0]
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

    # For each spatial coordinate in the weight's kernel, calculate how many
    # operations are performed every time that spatial coordinate is applied
    weight_spatial_flattened = numpy.reshape(weight, (*weight.shape[:2], -1))
    kernel_dense_sparse_ops = [
        _get_gemm_dense_sparse_ops(
            weight_spatial_flattened[:, :, i].T,
            [1, weight.shape[1]],
            zero_point,
            is_four_block_sparse,
        )
        for i in range(weight_spatial_flattened.shape[2])
    ]
    kernel_dense_sparse_ops = numpy.reshape(kernel_dense_sparse_ops, (*kernel_shape, 2))

    dense_sum, sparse_sum = 0, 0

    # TODO: This can be sped up by first calculating the number of operations
    #       performed with no padding, then adding additional operations for padding
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
    :return: number of dense and sparse operations performed
    """
    return (_numpy_prod_none_safe(output_shapes), 0)


def _numpy_prod_none_safe(input: Union[None, List[Union[None, int]]]) -> int:
    """
    :param input: list of integers to prod or None
    :return: result of numpy.prod ignoring Nones
    """
    if input is None:
        return 0

    _input = numpy.copy(input)
    _input[_input is None] = 1

    return numpy.prod(_input)
