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

from typing import Any, Dict, List, Union

import yaml
from onnx import NodeProto

from sparsezoo.analyze_v2.model_validator import (
    OperationAnalysisModel,
    QuantizationAnalysisModel,
    SparsityAnalysisModel,
)
from sparsezoo.utils import (
    ONNXGraph,
    get_node_weight,
    get_numpy_quantization_level,
    get_ops_count_from_ops_dict,
    get_ops_dict,
    is_four_block_sparse_layer,
    is_sparse_layer,
)


class OperationAnalysis:
    """
    Given model_graph and node, compute

    1. counts / sparsity level
    2. bits / quantization level

    with respect to the weight operations in the node.
    """

    def __init__(
        self,
        model_graph: ONNXGraph,
        node: NodeProto,
        node_shape,
    ):
        self.model_graph = model_graph
        self.node = node
        self.node_shape = node_shape

        self.counts: Dict = {}  # single grouping param counts
        self.bits: Dict = {}  # Tensor grouping bits

        self.sparsity_analysis_model = self.get_sparsity()
        self.quantization_analysis_model = self.get_quantization()

    def get_sparsity(self) -> List["SparsityAnalysisModel"]:
        """Get the number of operation counts"""

        data = get_operation_counts(self.model_graph, self.node, self.node_shape)
        sparsity_analysis_model = []
        for grouping, counts_dict in data.items():
            if grouping == "single":
                self.counts = counts_dict

            sparsity_analysis_model.append(
                SparsityAnalysisModel(grouping=grouping, **counts_dict)
            )

        return sparsity_analysis_model

    def get_quantization(self) -> List["QuantizationAnalysisModel"]:
        """Get the number of bits and quantized bits from weights"""
        data = get_operation_bits(self.model_graph, self.node, self.node_shape)
        quantization_analysis_model = []
        for grouping, counts_dict in data.items():
            if grouping == "tensor":
                self.bits = counts_dict

            quantization_analysis_model.append(
                QuantizationAnalysisModel(grouping=grouping, **counts_dict)
            )

        return quantization_analysis_model

    def to_dict(self) -> Dict[str, Any]:
        return OperationAnalysisModel(
            name=self.node.name,
            sparsity=self.sparsity_analysis_model,
            quantization=self.quantization_analysis_model,
        ).dict()

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict())


def get_operation_counts(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape,
) -> Dict[str, Union[int, float]]:
    """Get the number of operations for the weighted layers"""
    ops_sparse, ops_sparse_block_four = 0, 0

    ops_dict_single = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=False
    )
    ops_dict_block_four = get_ops_dict(
        model_graph, node, node_shape=node_shape, is_four_block_sparse=True
    )

    ops_dense = get_ops_count_from_ops_dict("num_dense_ops", ops_dict_single)
    ops_dense_block4 = get_ops_count_from_ops_dict("num_dense_ops", ops_dict_block_four)
    true_ops_dict = (
        ops_dict_single
        if not is_four_block_sparse_layer(model_graph, node)
        else ops_dict_block_four
    )

    if is_sparse_layer(model_graph, node):
        ops_sparse = get_ops_count_from_ops_dict("num_sparse_ops", true_ops_dict)
        ops_sparse_block_four = get_ops_count_from_ops_dict(
            "num_sparse_ops", ops_dict_block_four
        )

    ops_total = ops_dense + ops_sparse
    ops_total_block_four = ops_dense_block4 + ops_sparse_block_four
    return {
        "single": {
            "counts": ops_total,
            "counts_sparse": ops_sparse,
        },
        "block4": {
            "counts": ops_total_block_four,
            "counts_sparse": ops_sparse_block_four,
        },
    }


def get_operation_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shapes,
) -> Dict[str, Union[int, float]]:
    """Get the number of bits and quantized bits from ops"""
    bits, bits_block4 = 0, 0
    is_quantized_op = False

    node_weight = get_node_weight(model_graph, node)
    if node_weight is not None and node_weight.size > 0:

        precision = get_numpy_quantization_level(node_weight)
        is_quantized_op = "32" not in str(precision)

        ops = get_operation_counts(model_graph, node, node_shapes)

        bits = (ops["single"]["counts"] + ops["single"]["counts_sparse"]) * precision

        bits_block4 = (
            ops["block4"]["counts"] + ops["block4"]["counts_sparse"]
        ) * precision

    bits_quant = is_quantized_op * bits
    return {
        "tensor": {
            "bits": bits,
            "bits_quant": bits_quant,
        },
        "block4": {
            "bits": bits_block4,
            "bits_quant": bits_quant,
        },
    }
