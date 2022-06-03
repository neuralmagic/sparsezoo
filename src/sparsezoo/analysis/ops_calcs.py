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
    get_zero_point,
    get_node_num_four_block_zeros_and_size,
)


def get_num_dense_and_sparse_ops(model: ModelProto, node: NodeProto, node_shapes: Optional[Dict[str, NodeShape]] = None):
    if node_shapes is None:
        node_shapes = extract_node_shapes(model)
        raise Exception("TODO: Remove this exception")

    node_shape = node_shapes.get(extract_node_id(node))
    input_shapes = node_shape.input_shapes if node_shape is not None else None
    output_shapes = node_shape.output_shapes if node_shape is not None else None

    weight = get_node_weight(model, node)
    zero_point = get_zero_point(model, node)

    node_attributes = get_node_attributes(node)

    def _get_gemm_dense_sparse_ops(weight, input_shape, zero_point=0, is_four_block=False):
        if is_four_block:
            pass
        else:
            num_zeros = numpy.count_nonzero(weight == zero_point)
            num_non_zeros = numpy.count_nonzero(weight != zero_point)

        num_dense_ops = 2 * input_shape[0] * num_non_zeros
        num_sparse_ops = 2 * input_shape[0] * num_zeros

        if is_four_block:
            num_dense_ops *= 4
            num_sparse_ops *= 4

        return num_dense_ops, num_sparse_ops

    print(node)
    print(node.op_type)
    print(input_shapes)
    print(weight.shape)
    print(output_shapes)

    # FAKE FOURBLOCK PRUNE
    weight = numpy.array(weight)
    weight[0, 0] = zero_point
    weight[1, 0] = zero_point
    weight[2, 0] = zero_point
    weight[3, 0] = zero_point
    #print(get_node_num_four_block_zeros_and_size(model, node))

    if node.op_type == "Gemm":
        input_shape = input_shapes[0]

        return _get_gemm_dense_sparse_ops(weight, input_shape, zero_point)

#def calculate_gemm_ops(weight, input_shape):

#    return 2 *
