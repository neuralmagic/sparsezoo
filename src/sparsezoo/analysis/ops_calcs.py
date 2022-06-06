from typing import Dict, Optional, Tuple

import numpy
from onnx import ModelProto, NodeProto, numpy_helper

from sparsezoo.analysis.onnx_helpers import (
    NodeShape,
    calculate_num_operations,
    extract_node_id,
    extract_node_shapes,
    get_kernel_shape,
    get_node_attributes,
)

from sparsezoo.analysis.helpers import (
    get_node_weight,
    get_node_bias,
    get_zero_point,
    group_four_block,
    is_four_block_sparse_layer,
)

def _get_gemm_dense_sparse_ops(weight, input_shape, zero_point=0, is_four_block_sparse=False):
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

def _get_kernel_subview(weight, x, y, spatial_shape, kernel_shape, pads):
    distance_from_right = (spatial_shape[1] - x - 1)
    k_x_start = kernel_shape[1] - pads[2] - distance_from_right - 1
    k_x_start = max(k_x_start, 0)
    k_x_end   = kernel_shape[1] + pads[0] + x
    k_x_end   = min(k_x_end, kernel_shape[1])

    distance_from_bottom = (spatial_shape[0] - y - 1)
    k_y_start = kernel_shape[0] - pads[3] - distance_from_bottom - 1
    k_y_start = max(k_y_start, 0)
    k_y_end   = kernel_shape[0] + pads[1] + y
    k_y_end   = min(k_y_end, kernel_shape[0])

    return weight[:, :, k_y_start: k_y_end, k_x_start: k_x_end]


def _get_conv_weight_dense_sparse_ops(weight, input_shape, pads, strides, group, zero_point=0, is_four_block_sparse=False):
    spatial_shape = input_shape[2:]
    kernel_shape = weight.shape[2:]

    dense_sum, sparse_sum = 0, 0

    # For each pixel in the input
    for x in range(0, spatial_shape[1], strides[1]):
        for y in range(0, spatial_shape[0], strides[0]):

            # Calculate a subview of the kernel values that apply to this pixel
            sub_kernels = _get_kernel_subview(weight, x, y, spatial_shape, kernel_shape, pads)

            # Flatten kernel values [o_c, i_c, -1] where -1 is a list of spatial coordinate values
            sub_kernels_values = numpy.reshape(sub_kernels, (sub_kernels.shape[0], sub_kernels.shape[1], -1))

            # For each relevant kernel spatial coordinate, apply gemm across input channels
            for sub_kernels_value in sub_kernels_values:
                gemm_dense_ops, gemm_sparse_ops = _get_gemm_dense_sparse_ops(sub_kernels_value, [1, weight.shape[1]], zero_point, is_four_block_sparse)
                dense_sum += gemm_dense_ops
                sparse_sum += gemm_sparse_ops

    # Adjust for depthwise convolutions
    dense_sum = dense_sum // group
    sparse_sum = sparse_sum // group

    return dense_sum, sparse_sum


def _get_conv_bias_dense_sparse_ops(bias, output_shape, zero_point=0):
    output_spatial_shape = output_shape[-2:]

    num_zeros = numpy.count_nonzero(bias == zero_point)
    num_non_zeros = numpy.count_nonzero(bias != zero_point)

    num_dense_ops = num_non_zeros * numpy.prod(output_spatial_shape) * 2
    num_sparse_ops = num_zeros * numpy.prod(output_spatial_shape) * 2

    return num_dense_ops, num_sparse_ops


def _get_linear_bias_dense_sparse_ops(bias, zero_point=0):
    num_zeros = numpy.count_nonzero(bias == zero_point)
    num_non_zeros = numpy.count_nonzero(bias != zero_point)

    num_dense_ops = num_non_zeros * 2
    num_sparse_ops = num_zeros * 2

    return num_dense_ops, num_sparse_ops


def get_num_dense_and_sparse_ops(model: ModelProto, node: NodeProto, node_shapes: Optional[Dict[str, NodeShape]] = None):
    if node_shapes is None:
        node_shapes = extract_node_shapes(model)
        raise Exception("TODO: Remove this exception")

    node_shape = node_shapes.get(extract_node_id(node))
    input_shapes = node_shape.input_shapes if node_shape is not None else None
    output_shapes = node_shape.output_shapes if node_shape is not None else None

    weight = get_node_weight(model, node)
    bias = get_node_bias(model, node)
    zero_point = get_zero_point(model, node)
    is_four_block_sparse = is_four_block_sparse_layer(model, node)

    node_attributes = get_node_attributes(node)

    #if node.op_type in ["Add", "Mul", "Div", "Sub", "Clip"] + ["Relu", "LeakyRelu", "Sigmoid", "Tanh", "BatchNormalization"] + ["GlobalAveragePool", "GlobalMaxPool"]
    #if node.op_type in ["MaxPool", "AveragePool"]:

    print(node.op_type)
    if node.op_type in ["Gemm", "MatMul", "MatMulInteger", "QLinearMatMul"]:
        input_shape = input_shapes[0]

        # If no weight supplied, treat other input as dense weight
        if weight is None:
            weight_shape = input_shapes[1]
            weight = numpy.full(weight_shape, zero_point - 1)

        # Weight operations
        num_dense_ops, num_sparse_ops = _get_gemm_dense_sparse_ops(weight, input_shape, zero_point=zero_point, is_four_block_sparse=is_four_block_sparse)

        # Bias operations
        if not bias is None:
            bias_dense_ops, bias_sparse_ops = _get_linear_bias_dense_sparse_ops(bias, zero_point)
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
        num_dense_ops, num_sparse_ops = _get_conv_weight_dense_sparse_ops(weight, input_shape, pads, strides, group, zero_point=zero_point, is_four_block_sparse=is_four_block_sparse)

        # Bias operations
        if not bias is None:
            bias_dense_ops, bias_sparse_ops = _get_conv_bias_dense_sparse_ops(bias, output_shape, zero_point)
            num_dense_ops += bias_dense_ops
            num_sparse_ops += bias_sparse_ops

        return num_dense_ops, num_sparse_ops

    return None
