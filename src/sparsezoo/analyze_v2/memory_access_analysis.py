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

from functools import reduce
from typing import Any, Dict, List, Optional

import yaml
from onnx import NodeProto

from sparsezoo.analyze_v2.schemas import (
    MemoryAccessAnalysisSchema,
    QuantizationAnalysisSchema,
    SparsityAnalysisSchema,
)
from sparsezoo.utils import (
    ONNXGraph,
    get_node_kernel_shape,
    get_node_num_four_block_zeros_and_size,
    get_node_param_counts,
    get_node_weight,
    get_numpy_quantization_level,
    is_quantized_layer,
    is_sparse_layer,
)


class MemoryAccessAnalysis:
    def __init__(
        self,
        model_graph: ONNXGraph,
        node: NodeProto,
        node_shape: Dict,
    ):
        self.model_graph = model_graph
        self.node = node
        self.node_shape = node_shape

        self.sparsity_analysis_model = self.get_sparsity()
        self.quantization_analysis_model = self.get_quantization()

    def get_sparsity(self) -> List["SparsityAnalysisSchema"]:
        """
        Get the number of dense and sparse weights, if any

        :returns: List of sparsity analysis pydantic models for each grouping
         if the node has weights
        """
        data = get_memory_access_counts(self.model_graph, self.node, self.node_shape)
        if data is not None:
            sparsity_analysis_model = []
            for grouping, counts_dict in data.items():
                sparsity_analysis_model.append(
                    SparsityAnalysisSchema(grouping=grouping, **counts_dict)
                )

            return sparsity_analysis_model

    def get_quantization(self) -> List["QuantizationAnalysisSchema"]:
        """
        Get the number of bits and quantized bits from weights

        :returns: List of quantization analysis pydantic models for each grouping
         if the node has weights
        """
        data = get_memory_access_bits(self.model_graph, self.node, self.node_shape)
        if data is not None:
            quantization_analysis_model = []
            for grouping, counts_dict in data.items():
                quantization_analysis_model.append(
                    QuantizationAnalysisSchema(grouping=grouping, **counts_dict)
                )
            return quantization_analysis_model

    def to_dict(self) -> Dict[str, Any]:
        if self.sparsity_analysis_model:
            return MemoryAccessAnalysisSchema(
                name=self.node.name,
                sparsity=[dict(model) for model in self.sparsity_analysis_model],
                quantization=[
                    dict(model) for model in self.quantization_analysis_model
                ],
            ).dict()

    def to_yaml(self) -> str:
        if self.sparsity_analysis_model:
            return yaml.dump(self.to_dict())


def get_size_from_shape(arr: Optional[List] = None):
    if arr:
        return reduce(lambda el, res: el * res, arr)
    return 0


def get_memory_access_counts(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Dict[str, List],
):
    num_weights, _, num_weights_sparse = get_node_param_counts(node, model_graph)
    if num_weights > 0:
        out_feat_size = get_size_from_shape(node_shape.input_shapes[0])
        inp_feat_size = get_size_from_shape(node_shape.output_shapes[0])
        kernel_shape = get_node_kernel_shape(node)
        kernel_size = get_size_from_shape(kernel_shape) if kernel_shape else 0

        num_sparse_weights_four_blocks, num_weights_four_block = 0, 0
        if is_sparse_layer(model_graph, node):
            (
                num_sparse_weights_four_blocks,
                num_weights_four_block,
            ) = get_node_num_four_block_zeros_and_size(model_graph, node)

        counts = (
            num_weights * out_feat_size + inp_feat_size * kernel_size + out_feat_size
        )
        counts_sparse = (
            num_weights_sparse * out_feat_size
            + inp_feat_size * kernel_size
            + out_feat_size
        )
        counts_block4 = (
            num_weights_four_block * out_feat_size
            + inp_feat_size * kernel_size
            + out_feat_size
        )
        counts_block4_sparse = (
            num_sparse_weights_four_blocks * out_feat_size
            + inp_feat_size * kernel_size
            + out_feat_size
        )
        return {
            "single": {
                "counts": counts,
                "counts_sparse": counts_sparse,
            },
            "block4": {
                "counts": counts_block4,
                "counts_sparse": counts_block4_sparse,
            },
        }


def get_memory_access_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Dict,
):
    num_weights, _, _ = get_node_param_counts(node, model_graph)
    if num_weights > 0:
        memory_access_counts: Dict = get_memory_access_counts(
            model_graph, node, node_shape
        )
        node_weight = get_node_weight(model_graph, node)
        precision = get_numpy_quantization_level(node_weight)
        counts = memory_access_counts["single"]["counts"]
        bits = counts * precision
        is_quantized = is_quantized_layer(model_graph, node)

        return {
            "tensor": {
                "bits": bits,
                "bits_quant": bits * is_quantized,
                "counts": counts,
                "counts_quant": counts * is_quantized,
            }
        }
