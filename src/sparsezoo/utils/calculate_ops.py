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

import copy
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy
from onnx import NodeProto

from sparsezoo.utils import (
    ONNXGraph,
    get_node_attributes,
    get_node_bias,
    get_node_weight,
    get_zero_point,
    group_four_block,
    is_four_block_sparse_layer,
)
from sparsezoo.utils.node_inference import NodeShape


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "get_ops_dict",
]

EMPTY_OPS_DICT = {
    "weight": {"num_sparse_ops": 0, "num_dense_ops": 0},
    "bias": {"num_sparse_ops": 0, "num_dense_ops": 0},
    "other": {"num_sparse_ops": 0, "num_dense_ops": 0},
}


def get_ops_dict(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: NodeShape,
    is_four_block_sparse: Optional[bool] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Gets an approximation of the number of floating point or integer operations

    :param model_graph: model graph that contains the given node
    :param node: node which performs the operations
    :param node_shape: the shapes associated with this node
    :param is_four_block_sparse: optional boolean indicating if this node is four
        block sparse. If not supplied, it be will be computed
    :return: dictionary of counts with the following structure
        - weight:
            - num_sparse_ops
            - num_dense_ops
        - bias:
            - num_sparse_ops
            - num_dense_ops
        - other:
            - num_sparse_ops
            - num_dense_ops
    """
    input_shapes = node_shape.input_shapes if node_shape is not None else None
    output_shapes = node_shape.output_shapes if node_shape is not None else None

    weight = get_node_weight(model_graph, node)
    bias = get_node_bias(model_graph, node)
    zero_point = get_zero_point(model_graph, node)
    is_four_block_sparse = (
        is_four_block_sparse_layer(model_graph, node)
        if is_four_block_sparse is None
        else is_four_block_sparse
    )
    node_attributes = get_node_attributes(node)

    ops_dict = copy.deepcopy(EMPTY_OPS_DICT)

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
        ops_dict["other"]["num_dense_ops"] = _numpy_prod_none_safe(output_shapes)

    # If BN is followed by matmul or conv, then it is folded into the following
    # layer weights. Assume this is true for all BN cases
    if node.op_type == "BatchNormalization":
        pass

    if node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
        ops_dict["other"]["num_dense_ops"] = _numpy_prod_none_safe(input_shapes)

    if node.op_type in ["MaxPool", "AveragePool"]:
        kernel_shape = node_attributes["kernel_shape"]
        ops_dict["other"]["num_dense_ops"] = _numpy_prod_none_safe(
            output_shapes
        ) * _numpy_prod_none_safe(kernel_shape)

    if node.op_type in ["Gemm", "MatMul", "MatMulInteger", "QLinearMatMul"]:
        if input_shapes is None:
            _LOGGER.warn(
                "Invalid shape, skipping "
                f"{'four block ' if is_four_block_sparse else ''}ops calculation"
                f" for {node.name}"
            )
            return ops_dict

        input_shape = input_shapes[0]

        # If no weight supplied, treat other input as dense weight
        if weight is None:
            weight_shape = input_shapes[1]
            weight = numpy.full(weight_shape, zero_point - 1)

        # Weight operations
        num_dense_weight_ops, num_sparse_weight_ops = _get_gemm_dense_sparse_ops(
            weight,
            input_shape,
            zero_point=zero_point,
            is_four_block_sparse=is_four_block_sparse,
        )
        ops_dict["weight"]["num_dense_ops"] = num_dense_weight_ops
        ops_dict["weight"]["num_sparse_ops"] = num_sparse_weight_ops

        # Bias operations
        if bias is not None:
            bias_dense_ops, bias_sparse_ops = _get_bias_dense_sparse_ops(output_shapes)
            ops_dict["bias"]["num_dense_ops"] = bias_dense_ops
            ops_dict["bias"]["num_sparse_ops"] = bias_sparse_ops

    if node.op_type in ["Conv", "ConvInteger", "QLinearConv"]:
        if input_shapes is None:
            _LOGGER.warn(
                "Invalid shape, skipping "
                f"{'four block ' if is_four_block_sparse else ''}ops calculation"
                f" for {node.name}"
            )
            return ops_dict

        input_shape = input_shapes[0]
        pads = node_attributes["pads"] if "pads" in node_attributes else [0, 0, 0, 0]
        strides = node_attributes["strides"] if "strides" in node_attributes else [1, 1]
        group = node_attributes["group"] if "group" in node_attributes else 1

        # Weight operations
        num_dense_weight_ops, num_sparse_weight_ops = _get_conv_weight_dense_sparse_ops(
            weight,
            input_shape,
            pads,
            strides,
            group,
            zero_point=zero_point,
            is_four_block_sparse=is_four_block_sparse,
        )
        ops_dict["weight"]["num_dense_ops"] = num_dense_weight_ops
        ops_dict["weight"]["num_sparse_ops"] = num_sparse_weight_ops

        # Bias operations
        if bias is not None:
            bias_dense_ops, bias_sparse_ops = _get_bias_dense_sparse_ops(output_shapes)
            ops_dict["bias"]["num_dense_ops"] = bias_dense_ops
            ops_dict["bias"]["num_sparse_ops"] = bias_sparse_ops

    return ops_dict


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
            weight=weight_spatial_flattened[:, :, i].T,
            input_shape=[1, weight.shape[1]],
            zero_point=zero_point,
            is_four_block_sparse=is_four_block_sparse,
        )
        for i in range(weight_spatial_flattened.shape[2])
    ]
    kernel_dense_sparse_ops = numpy.reshape(kernel_dense_sparse_ops, (*kernel_shape, 2))

    dense_sum, sparse_sum = 0, 0

    # area covered by each kernel position
    kernel_receptive_field_shape = deepcopy(spatial_shape)
    kernel_receptive_field_shape[1] -= kernel_shape[1] - 1
    kernel_receptive_field_shape[0] -= kernel_shape[0] - 1

    for kernel_x in range(kernel_shape[1]):
        for kernel_y in range(kernel_shape[0]):
            from_left = kernel_x
            from_right = spatial_shape[1] + kernel_shape[1] - kernel_x - 3
            from_top = kernel_y
            from_bottom = spatial_shape[0] + kernel_shape[0] - kernel_y - 3

            # calculate receptive field for each kernel coordinate
            coord_receptive_field_shape = deepcopy(kernel_receptive_field_shape)

            # expand area by pads[top, left, bottom, right]
            coord_receptive_field_shape[1] += max(min(pads[3], from_right), 0)
            coord_receptive_field_shape[1] += max(min(pads[1], from_left), 0)
            coord_receptive_field_shape[0] += max(min(pads[0], from_top), 0)
            coord_receptive_field_shape[0] += max(min(pads[2], from_bottom), 0)

            # scale area by strides
            coord_receptive_field_shape[1] = int(
                numpy.ceil(coord_receptive_field_shape[1] / strides[1])
            )
            coord_receptive_field_shape[0] = int(
                numpy.ceil(coord_receptive_field_shape[0] / strides[0])
            )

            # accumulate
            dense_sum += kernel_dense_sparse_ops[kernel_y, kernel_x, 0] * numpy.prod(
                coord_receptive_field_shape
            )
            sparse_sum += kernel_dense_sparse_ops[kernel_y, kernel_x, 1] * numpy.prod(
                coord_receptive_field_shape
            )

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
    _input[_input == None] = 1  # noqa: E711

    return numpy.prod(_input)
