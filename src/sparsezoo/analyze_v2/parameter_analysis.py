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

from typing import Any, Dict, List, Optional, Union

import numpy
import yaml
from onnx import NodeProto

from sparsezoo.analyze_v2.schemas import (
    DistributionAnalysisSchema,
    ParameterAnalysisSchema,
    QuantizationAnalysisSchema,
    SparsityAnalysisSchema,
)
from sparsezoo.utils import (
    ONNXGraph,
    get_node_num_four_block_zeros_and_size,
    get_node_param_counts,
    get_node_weight,
    get_node_weight_bits,
    get_numpy_distribution_statistics,
    get_numpy_entropy,
    get_numpy_modes,
    get_numpy_percentiles,
    is_quantized_layer,
)


class ParameterAnalysis:
    """
    Given model_graph and node, compute

    1. counts / sparsity level
    2. bits / quantization level
    3. distribution

    with respect to the weights in the node, if any.
    """

    def __init__(
        self,
        model_graph: ONNXGraph,
        node: NodeProto,
    ):
        self.model_graph = model_graph
        self.node = node

        self.sparsity_analysis_model = self.get_sparsity()
        self.quantization_analysis_model = self.get_quantization()
        self.distribution_model = self.get_distribution()

    def get_sparsity(self) -> Optional[List["SparsityAnalysisSchema"]]:
        """
        Get the number of dense and sparse weights, if any

        :returns: List of sparsity analysis pydantic models for each grouping
         if the node has weights
        """

        data = get_parameter_counts(self.model_graph, self.node)
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
        data = get_parameter_bits(self.model_graph, self.node)
        if data is not None:
            quantization_analysis_model = []
            for grouping, counts_dict in data.items():
                quantization_analysis_model.append(
                    QuantizationAnalysisSchema(grouping=grouping, **counts_dict)
                )

            return quantization_analysis_model

    def get_distribution(self) -> Optional["DistributionAnalysisSchema"]:
        """Get the distribution statistics with respect to the weights"""
        distribution_dct = get_parameter_distribution(self.model_graph, self.node)
        if distribution_dct:
            return DistributionAnalysisSchema(**distribution_dct)

    def to_dict(self) -> Optional[Dict[str, Any]]:
        if self.sparsity_analysis_model:
            return ParameterAnalysisSchema(
                name=self.node.name,
                op_type=self.node.op_type,
                distribution=self.distribution_model,
                sparsity=self.sparsity_analysis_model,
                quantization=self.quantization_analysis_model,
            ).dict()

    def to_yaml(self) -> Optional[str]:
        if self.sparsity_analysis_model:
            return yaml.dump(self.to_dict())


def get_parameter_counts(
    model_graph: ONNXGraph,
    node: NodeProto,
) -> Optional[Dict[str, Union[int, float]]]:
    """Get the number of parameters in the node, if any"""

    num_weights, _, num_weights_sparse = get_node_param_counts(node, model_graph)
    if num_weights > 0:
        (
            num_sparse_weights_four_blocks,
            num_weights_four_block,
        ) = get_node_num_four_block_zeros_and_size(model_graph, node)

        return {
            "single": {
                "counts": num_weights,
                "counts_sparse": num_weights_sparse,
            },
            "block4": {
                "counts": num_weights_four_block,
                "counts_sparse": num_sparse_weights_four_blocks,
            },
        }


def get_parameter_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
    *args,
    **kwargs,
) -> Optional[Dict[str, Union[int, float]]]:
    """
    Get the number of bits used to store the array
    If the layer is quantized, assume all its elements in the ndarray
     are quantized
    """
    node_weight = get_node_weight(model_graph, node)
    if node_weight is not None and node_weight.size > 0:
        bits = get_node_weight_bits(model_graph, node)

        return {
            "tensor": {
                "bits": bits,
                "bits_quant": bits * is_quantized_layer(model_graph, node),
            },
        }


def get_parameter_distribution(
    model_graph: ONNXGraph,
    node: NodeProto,
    num_bins: int = 25,
    *args,
    **kwargs,
) -> Optional[Dict[str, Union[int, float]]]:
    """Get the statistics of the parameters in the node if any"""

    node_weight = get_node_weight(model_graph, node)

    if node_weight is not None and node_weight.size > 0:
        mean = node_weight.mean()
        counts = node_weight.size
        median = numpy.median(node_weight)
        modes = get_numpy_modes(node_weight)
        sum_val = numpy.sum(node_weight)
        min_val = numpy.min(node_weight)
        max_val = numpy.max(node_weight)
        percentiles = get_numpy_percentiles(node_weight)

        std_dev = numpy.std(node_weight)
        skewness, kurtosis = get_numpy_distribution_statistics(node_weight)
        entropy = get_numpy_entropy(node_weight)

        bin_width = (max_val - min_val) / num_bins
        hist, bin_edges = numpy.histogram(node_weight, bins=num_bins)

        return {
            "counts": counts,
            "mean": mean,
            "median": median,
            "modes": modes,
            "sum_val": sum_val,
            "min_val": min_val,
            "max_val": max_val,
            "percentiles": percentiles,
            "std_dev": std_dev,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "entropy": entropy,
            "num_bins": num_bins,
            "bin_width": bin_width,
            "hist": hist.tolist() if isinstance(hist, numpy.ndarray) else hist,
            "bin_edges": bin_edges.tolist()
            if isinstance(bin_edges, numpy.ndarray)
            else bin_edges,
        }
