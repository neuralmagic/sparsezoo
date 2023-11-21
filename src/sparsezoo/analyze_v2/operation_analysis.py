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

import numpy
import yaml
from onnx import ModelProto, NodeProto

# from sparsezoo._analysis.models import (
#     DistributionAnalysisModel,
#     ParamAnalysisModel,
#     QuantizationAnalysisModel,
#     SparsityAnalysisModel,
# )
from sparsezoo.utils import (  # get_node_ops,
    NodeDataType,
    NodeShape,
    ONNXGraph,
    extract_node_id,
    extract_node_shapes_and_dtypes,
    get_node_bias,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_param_counts,
    get_node_weight,
    get_numpy_bits,
    get_numpy_distribution_statistics,
    get_numpy_entropy,
    get_numpy_modes,
    get_numpy_percentiles,
    get_numpy_quantization_level,
    get_ops_count_from_ops_dict,
    get_ops_dict,
    is_four_block_sparse_layer,
    is_quantized_layer,
    is_sparse_layer,
    is_weighted_layer,
)


class OperationAnalysis:
    def __init__(
        self,
        model_graph: ONNXGraph,
        node: NodeProto,
        node_shape,
    ):
        self.model_graph = model_graph
        self.node = node
        self.node_shape = node_shape

        self.counts = self.get_counts()
        self.bits = self.get_bits()

    def get_counts(self):
        data = get_operation_counts(self.model_graph, self.node, self.node_shape)
        return {
            grouping: dict(
                percent=counts_dict["counts_sparse"] / counts_dict["counts"]
                if counts_dict["counts"] > 0
                else 0,
                counts=counts_dict["counts"],
                counts_sparse=counts_dict["counts_sparse"],
            )
            for grouping, counts_dict in data.items()
        }

    def get_bits(self):
        """
        Saves raw (tensor) and channel-wise metadata and
        returns parameter percentage of raw quantized params
        """
        data = get_operation_bits(self.model_graph, self.node, self.node_shape)
        return {
            grouping: dict(
                percent=quant_dict["bits_quant"] / quant_dict["bits"]
                if quant_dict["bits"] > 0
                else 0,
                bits_quant=quant_dict["bits_quant"],
                bits=quant_dict["bits"],
            )
            for grouping, quant_dict in data.items()
        }

    def to_dict(self):
        return dict(
            name=self.node.name,
            op_type=self.node.op_type,
            sparsity=self.counts,
            quantization=self.quantization,
        )

    def to_yaml(self):
        return yaml.dumps(self.to_dict())


def get_operation_counts(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape,
):
    single_ops_dict = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=False
    )
    four_block_ops_dict = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=True
    )

    ops_dense = get_ops_count_from_ops_dict("num_dense_ops", single_ops_dict)
    ops_dense_block4 = get_ops_count_from_ops_dict("num_dense_ops", four_block_ops_dict)
    true_ops_dict = (
        single_ops_dict
        if not is_four_block_sparse_layer(model_graph, node)
        else four_block_ops_dict
    )
    ops_sparse, ops_sparse_block4 = 0, 0

    if is_sparse_layer(model_graph, node):
        ops_sparse = get_ops_count_from_ops_dict("num_sparse_ops", true_ops_dict)
        ops_sparse_block4 = get_ops_count_from_ops_dict(
            "num_sparse_ops", four_block_ops_dict
        )
    # breakpoint()
    return {
        "single": {
            "counts": ops_dense + ops_sparse,
            "counts_sparse": ops_sparse,
        },
        "block4": {
            "counts": ops_dense_block4 + ops_sparse_block4,
            "counts_sparse": ops_sparse_block4,
        },
    }


def get_operation_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shapes,
):
    bits, bits_block4 = 0, 0
    is_quantized_op = False
    if is_weighted_layer(node):
        node_weight = get_node_weight(model_graph, node)

        precision = get_numpy_quantization_level(node_weight)
        is_quantized_op = "32" not in str(precision)

        ops = get_operation_counts(model_graph, node, node_shapes)

        bits = (ops["single"]["counts"] + ops["single"]["counts_sparse"]) * precision

        bits_block4 = (
            ops["block4"]["counts"] + ops["block4"]["counts_sparse"]
        ) * precision

        # if not is_quantized_op: breakpoint()

    return {
        "tensor": {
            "bits": bits,
            "bits_quant": is_quantized_op * bits,
        },
        "block4": {
            "bits": bits_block4,
            "bits_quant": is_quantized_op * bits_block4,
        },
    }
