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
from typing import Dict, List

import numpy
import yaml
from onnx import ModelProto, NodeProto

# from sparsezoo._analysis.models import (
#     DistributionAnalysisModel,
#     ParamAnalysisModel,
#     QuantizationAnalysisModel,
#     SparsityAnalysisModel,
# )
from sparsezoo.utils import (  # get_node_ops,; get_node_quantization,
    NodeDataType,
    NodeShape,
    ONNXGraph,
    extract_node_id,
    extract_node_shapes_and_dtypes,
    get_node_bias,
    get_node_input_feature_name,
    get_node_kernel_shape,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_output_name,
    get_node_param_counts,
    get_node_weight,
    get_node_weight_name,
    get_numpy_bits,
    get_numpy_distribution_statistics,
    get_numpy_entropy,
    get_numpy_modes,
    get_numpy_percentiles,
    get_numpy_quantization_level,
    get_ops_dict,
    is_four_block_sparse_layer,
    is_quantized_layer,
    is_sparse_layer,
    is_weighted_layer,
)
from sparsezoo.utils.node_inference import extract_nodes_shapes_and_dtypes_ort


class MemoryAccessAnalysis:
    def __init__(
        self,
        model_graph: ONNXGraph,
        node: NodeProto,
        node_shape: Dict,
        node_dtype: Dict,
    ):
        self.model_graph = model_graph
        self.node = node
        self.node_shape = node_shape
        self.node_dtype = node_dtype

        self.counts = self.get_counts()
        self.bits = self.get_bits()

    def get_counts(self):
        data = get_memory_access_counts(self.model_graph, self.node, self.node_shape)

        return {
            grouping: dict(
                percent=counts_dict["counts_sparse"] / counts_dict["counts"]
                if counts_dict["counts"] > 0
                else 0,
                counts_sparse=counts_dict["counts_sparse"],
                counts=counts_dict["counts"],
            )
            for grouping, counts_dict in data.items()
        }

    def get_bits(self):
        """
        Saves raw (tensor) and channel-wise metadata and
        returns parameter percentage of raw quantized params
        """
        data = get_memeory_access_bits(
            self.model_graph, self.node, self.node_shape, self.node_dtype
        )
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


def get_size_from_shape(arr: List):
    if arr:
        return reduce(lambda el, res: el * res, arr)
    return 0


def get_memeory_access_params(
    model_graph: ONNXGraph,
    node: NodeProto,
    graph_shapes: Dict,
):
    num_weights, out_feat_size, inp_feat_size, kernel = 0, 0, 0, 0
    num_weights_sparse = 0

    if is_weighted_layer(node):
        num_weights, num_bias, num_weights_sparse = get_node_param_counts(
            node, model_graph
        )
        out_feat_size = get_size_from_shape(graph_shapes[node.output[0]])
        inp_feat_size - get_size_from_shape(graph_shapes[node.input[0]])
        kernel = get_size_from_shape(get_node_kernel_shape(node))

        if is_sparse_layer(model_graph, node):
            (
                num_sparse_weights_four_blocks,
                num_weights_four_block,
            ) = get_node_num_four_block_zeros_and_size(model_graph, node)

    return num_weights, out_feat_size, inp_feat_size, kernel, num_weights_sparse


def get_memory_access_counts(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Dict,
):
    num_weights, out_feat_size, inp_feat_size, kernel = 0, 0, 0, 0
    num_weights_sparse, out_feat_size, inp_feat_size, kernel = 0, 0, 0, 0
    num_weights_four_block, num_sparse_weights_four_blocks = 0, 0

    if is_weighted_layer(node):
        num_weights, num_bias, num_weights_sparse = get_node_param_counts(
            node, model_graph
        )
        # if node.name == "Gemm_1239": breakpoint()
        out_feat_size = get_size_from_shape(node_shape.input_shapes[0])
        inp_feat_size - get_size_from_shape(node_shape.output_shapes[0])
        kernel_shape = get_node_kernel_shape(node)
        kernel = get_size_from_shape(kernel_shape) if kernel_shape else 0

        if is_sparse_layer(model_graph, node):
            (
                num_sparse_weights_four_blocks,
                num_weights_four_block,
            ) = get_node_num_four_block_zeros_and_size(model_graph, node)

    return {
        "single": {
            "counts": (
                num_weights * out_feat_size + inp_feat_size * kernel + out_feat_size
            ),
            "counts_sparse": (
                num_weights_sparse * out_feat_size
                + inp_feat_size * kernel
                + out_feat_size
            ),
        },
        "block4": {
            "counts": (
                num_weights_four_block * out_feat_size
                + inp_feat_size * kernel
                + out_feat_size
            ),
            "counts_sparse": (
                num_sparse_weights_four_blocks * out_feat_size
                + inp_feat_size * kernel
                + out_feat_size
            ),
        },
    }


def get_memeory_access_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Dict,
    node_dtype: Dict,
):
    bits, bits_quant = 0, 0
    if is_weighted_layer(node):
        dct = get_memory_access_counts(model_graph, node, node_shape)
        node_weight = get_node_weight(model_graph, node)
        precision = get_numpy_quantization_level(node_weight)
        bits = dct["single"]["counts"] * precision
        bits_quant = bits * is_quantized_layer(model_graph, node)
        # return {
        #     key: {
        #         "bits": value["counts"] * precision,
        #         "bits_quant": value["counts"] * precision * is_quantized_layer(model_graph, node),
        #     }
        #     for key, value in dct.items()
        # }
    return {
        "tensor": {
            "bits": bits,
            "bits_quant": bits_quant,
        }
    }


"""
def get_memeory_access_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Dict,
    node_dtype: Dict,
):
    bits, bits_quant = 0,0 
    if is_weighted_layer(node):
        dct = get_memory_access_counts(model_graph, node, node_shape)
        node_weight = get_node_weight(model_graph, node)
        precision = get_numpy_quantization_level(node_weight)
        bits = precision["counts"] * precision
        bits_quant = precision["counts"] * precision* is_quantized_layer(model_graph, node),
        
    return {
      "tensor": {
          "bits": bits,
          "bits_quant": bits_quant,
      }
    }
"""
