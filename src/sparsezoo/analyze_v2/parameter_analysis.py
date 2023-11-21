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

from typing import Dict

import numpy
import yaml
from onnx import ModelProto, NodeProto

from sparsezoo.utils import (  # get_node_ops,; get_node_quantization,
    NodeDataType,
    NodeShape,
    ONNXGraph,
    extract_node_id,
    extract_node_shapes_and_dtypes,
    get_node_bias,
    get_node_bits,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_param_counts,
    get_node_weight,
    get_numpy_bits,
    get_numpy_distribution_statistics,
    get_numpy_entropy,
    get_numpy_modes,
    get_numpy_percentiles,
    get_ops_dict,
    is_four_block_sparse_layer,
    is_quantized_layer,
    is_sparse_layer,
    is_weighted_layer,
)


class ParameterAnalysis:
    def __init__(
        self,
        model_graph: ONNXGraph,
        node: NodeProto,
    ):
        self.model_graph = model_graph
        self.node = node

        self.counts: Dict = self.get_counts()
        self.bits: Dict = self.get_bits()
        self.distribution: Dict = self.get_distribution()

    def get_counts(self):
        """Get the total number of weights that are dense and sparsified"""
        data = get_parameter_counts(self.model_graph, self.node)
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
        """Get the total number of bits and quantized bits from weights"""
        data = get_parameter_bits(self.model_graph, self.node)

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

    def get_distribution(self):
        return get_parameter_distribution(self.model_graph, self.node)

    def to_dict(self):
        return dict(
            name=self.node.name,
            op_type=self.node.op_type,
            distribution=self.distribution,
            sparsity=self.counts,
            quantization=self.bits,
        )

    def to_yaml(self):
        return yaml.dumps(self.to_dict())


def get_parameter_counts(
    model_graph: ONNXGraph,
    node: NodeProto,
):
    num_weights, num_bias, num_sparse_weights = 0, 0, 0
    num_sparse_weights_four_blocks, num_weights_four_block = 0, 0

    if is_weighted_layer(node):
        num_weights, num_bias, num_sparse_weights = get_node_param_counts(
            node, model_graph
        )
        num_sparse_weights_four_blocks, num_weights_four_block = 0, 0
        if is_sparse_layer(model_graph, node):
            (
                num_sparse_weights_four_blocks,
                num_weights_four_block,
            ) = get_node_num_four_block_zeros_and_size(model_graph, node)

    return {
        "single": {
            "counts": num_weights,
            "counts_sparse": num_sparse_weights,
        },
        "block4": {
            "counts": num_weights_four_block,
            "counts_sparse": num_sparse_weights_four_blocks,
        },
    }


def get_parameter_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
):
    bits = 0
    if is_weighted_layer(node):
        bits = get_node_bits(model_graph, node)

    return {
        "tensor": {
            "bits": bits,
            "bits_quant": bits * is_quantized_layer(model_graph, node),
        },
        # TODO: Channels
        #  "Channel": {
        #     "bits": bits_channel,
        #     "quantized_bits": quantized_bits_channel,
        # },
    }


def get_parameter_distribution(
    model_graph: ONNXGraph,
    node: NodeProto,
    num_bins: int = 25,
):
    counts, mean, median, modes, sum_val, min_val, max_val = 0, 0, 0, 0, 0, 0, 0
    percentiles, std_dev, skewness, kurtosis, entropy = 0, 0, 0, 0, 0
    bin_width, hist, bin_edges = 0, 0, 0
    if is_weighted_layer(node):
        node_weight = get_node_weight(model_graph, node)
        node_bias = get_node_bias(model_graph, node)

        if node_weight.size > 0:

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
        "hist": hist,
        "bin_edges": bin_edges,
    }
