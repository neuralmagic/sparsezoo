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


from typing import Optional
import copy

from onnx import ModelProto, NodeProto
import numpy


from sparsezoo.analyze.utils.models import NodeIO, DenseSparseOps, ZeroNonZeroParams

from sparsezoo.utils import (
    NodeDataType,
    NodeShape,
    ONNXGraph,
    get_ops_dict,
    is_four_block_sparse_layer,
    get_node_weight,
    get_node_bias,
    get_node_num_zeros_and_size,
)


def _sum_across_keys(dict, key):
    return sum([dict[k][key] for k in dict.keys()])


def get_distribution_analysis(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Optional[NodeShape] = None,
    node_dtype: Optional[NodeDataType] = None,
):
    # ----- OP COUNTS
    single_ops_dict = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=False
    )
    is_four_block_sparse = is_four_block_sparse_layer(model_graph, node)
    four_block_ops_dict = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=True
    )
    true_ops_dict = single_ops_dict if not is_four_block_sparse else four_block_ops_dict
    counts = (
        _sum_across_keys(true_ops_dict, "num_dense_ops")
        + _sum_across_keys(true_ops_dict, "num_sparse_ops"),
    )

    ...


def get_parameter_analysis(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Optional[NodeShape] = None,
    node_dtype: Optional[NodeDataType] = None,
):
    # ================= COUNTS ===================
    num_sparse_parameters, num_parameters = get_node_num_zeros_and_size(
        model_graph, node
    )
    node_bias = get_node_bias(model_graph, node)
    node_bias_size = node_bias.size if node_bias is not None else 0
    counts = num_parameters + node_bias_size

    # ================= SPARSITY ANALYSIS ===================
    percent = num_sparse_parameters / (counts + 0.1)
    grouping: str = None

    # ================= QUANTIZATION ANALYSIS ===================
    node_weight = get_node_weight(model_graph, node)
    param_dtypes = [
        str(param.dtype) for param in [node_weight, node_bias] if param is not None
    ]
    params = []
    for dtype in param_dtypes:
        params.append(
            ZeroNonZeroParams(
                zero=(
                    num_sparse_parameters
                    if node_weight is not None and str(node_weight.dtype) == dtype
                    else 0
                )
                + (
                    node_bias_size - numpy.count_nonzero(node_bias)
                    if node_bias is not None and str(node_bias.dtype) == dtype
                    else 0
                ),
                non_zero=(
                    num_parameters - num_sparse_parameters
                    if node_weight is not None and str(node_weight.dtype) == dtype
                    else 0
                )
                + (
                    numpy.count_nonzero(node_bias)
                    if node_bias is not None and str(node_bias.dtype) == dtype
                    else 0
                ),
            )
        )
    breakpoint()


def get_operation_analysis(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Optional[NodeShape] = None,
    node_dtype: Optional[NodeDataType] = None,
):
    name = node.name
    op_type = node.op_type

    # ================= COUNTS ===================
    single_ops_dict = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=False
    )
    is_four_block_sparse = is_four_block_sparse_layer(model_graph, node)
    four_block_ops_dict = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=True
    )
    true_ops_dict = single_ops_dict if not is_four_block_sparse else four_block_ops_dict
    counts = _sum_across_keys(true_ops_dict, "num_dense_ops") + _sum_across_keys(
        true_ops_dict, "num_sparse_ops"
    )

    # ================= SPARSITY ANALYSIS ===================

    sparse_counts = _sum_across_keys(true_ops_dict, "num_sparse_ops")
    percent = sparse_counts / (counts + 0.1)
    grouping: str = None

    # ================= QUANTIZATION ANALYSIS ===================

    has_output_shapes = node_shape is not None and node_shape.output_shapes is not None
    has_output_dtypes = node_dtype is not None and node_dtype.output_dtypes is not None

    outputs = (
        [
            NodeIO(name=name, shape=shape, dtype=str(dtype))
            for name, shape, dtype in zip(
                node.output, node_shape.output_shapes, node_dtype.output_dtypes
            )
        ]
        if has_output_shapes and has_output_dtypes
        else []
    )
    first_output = next(iter(outputs), None)

    other_op_dtype = first_output.dtype if first_output is not None else "unknown"

    node_weight = get_node_weight(model_graph, node)
    node_bias = get_node_bias(model_graph, node)

    param_dtypes = [
        str(param.dtype) for param in [node_weight, node_bias] if param is not None
    ]
    operation_dtypes = copy.deepcopy(param_dtypes)

    ops = []
    for dtype in operation_dtypes:
        ops.append(
            DenseSparseOps(
                dense=(
                    (
                        true_ops_dict["weight"]["num_dense_ops"]
                        if node_weight is not None and str(node_weight.dtype) == dtype
                        else 0
                    )
                    + (
                        true_ops_dict["bias"]["num_dense_ops"]
                        if node_bias is not None and str(node_bias.dtype) == dtype
                        else 0
                    )
                    + (
                        true_ops_dict["other"]["num_dense_ops"]
                        if other_op_dtype == dtype
                        else 0
                    )
                ),
                sparse=(
                    (
                        true_ops_dict["weight"]["num_sparse_ops"]
                        if node_weight is not None and str(node_weight.dtype) == dtype
                        else 0
                    )
                    + (
                        true_ops_dict["bias"]["num_sparse_ops"]
                        if node_bias is not None and str(node_bias.dtype) == dtype
                        else 0
                    )
                    + (
                        true_ops_dict["other"]["num_sparse_ops"]
                        if other_op_dtype == dtype
                        else 0
                    )
                ),
            )
        )
