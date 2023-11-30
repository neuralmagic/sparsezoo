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
from typing import Dict, List, Optional

import yaml
from onnx import NodeProto

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

        self.counts = self.get_counts()
        self.bits = self.get_bits()

    def get_counts(self):
        """Get the numeber of times"""
        data = get_memory_access_counts(self.model_graph, self.node, self.node_shape)

        return {
            grouping: dict(
                counts=counts_dict["counts"],
                counts_sparse=counts_dict["counts_sparse"],
                percent=counts_dict["counts_sparse"] / counts_dict["counts"]
                if counts_dict["counts"] > 0
                else 0,
            )
            for grouping, counts_dict in data.items()
        }

    def get_bits(self):
        """
        Saves raw (tensor) and channel-wise metadata and
        returns parameter percentage of raw quantized params
        """
        data = get_memeory_access_bits(
            self.model_graph,
            self.node,
            self.node_shape,
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

    def to_dict(self):
        return dict(
            name=self.node.name,
            op_type=self.node.op_type,
            sparsity=self.counts,
            quantization=self.bits,
        )

    def to_yaml(self):
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
    out_feat_size, inp_feat_size, kernel_size = 0, 0, 0
    num_weights_four_block, num_sparse_weights_four_blocks = 0, 0

    num_weights, _, num_weights_sparse = get_node_param_counts(node, model_graph)

    if num_weights > 0:
        out_feat_size = get_size_from_shape(node_shape.input_shapes[0])
        inp_feat_size - get_size_from_shape(node_shape.output_shapes[0])
        kernel_shape = get_node_kernel_shape(node)
        kernel_size = get_size_from_shape(kernel_shape) if kernel_shape else 0

        if is_sparse_layer(model_graph, node):
            (
                num_sparse_weights_four_blocks,
                num_weights_four_block,
            ) = get_node_num_four_block_zeros_and_size(model_graph, node)

    return {
        "single": {
            "counts": (
                num_weights * out_feat_size
                + inp_feat_size * kernel_size
                + out_feat_size
            ),
            "counts_sparse": (
                num_weights_sparse * out_feat_size
                + inp_feat_size * kernel_size
                + out_feat_size
            ),
        },
        "block4": {
            "counts": (
                num_weights_four_block * out_feat_size
                + inp_feat_size * kernel_size
                + out_feat_size
            ),
            "counts_sparse": (
                num_sparse_weights_four_blocks * out_feat_size
                + inp_feat_size * kernel_size
                + out_feat_size
            ),
        },
    }


def get_memeory_access_bits(
    model_graph: ONNXGraph,
    node: NodeProto,
    node_shape: Dict,
):
    bits, bits_quant = 0, 0
    num_weights, _, _ = get_node_param_counts(node, model_graph)
    if num_weights > 0:
        dct = get_memory_access_counts(model_graph, node, node_shape)
        node_weight = get_node_weight(model_graph, node)
        precision = get_numpy_quantization_level(node_weight)
        bits = dct["single"]["counts"] * precision
        bits_quant = bits * is_quantized_layer(model_graph, node)

    return {
        "tensor": {
            "bits": bits,
            "bits_quant": bits_quant,
        }
        # TODO: Channel wise quantization
    }
