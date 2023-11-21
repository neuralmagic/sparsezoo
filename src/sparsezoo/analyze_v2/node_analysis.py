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
from onnx import ModelProto, NodeProto

from sparsezoo.utils import (  # get_node_ops,; get_node_quantization,
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
    get_ops_dict,
    is_four_block_sparse_layer,
    is_quantized_layer,
    is_sparse_layer,
)
from sparsezoo.utils.node_inference import extract_nodes_shapes_and_dtypes_ort

from .memory_access_analysis import MemoryAccessAnalysis
from .operation_analysis import OperationAnalysis
from .parameter_analysis import ParameterAnalysis


class NodeAnalysis:
    def __init__(self, model_graph, node, node_shape, node_dtype):
        self.node = node
        self.model_graph = model_graph

        self.node_shape = node_shape
        self.node_dtype = node_dtype
        # self.has_weights = False
        self.parameter_analysis = ParameterAnalysis(model_graph, node)
        self.operation_analysis = OperationAnalysis(model_graph, node, node_shape)
        self.memory_access_analysis = MemoryAccessAnalysis(
            model_graph, node, node_shape, node_dtype
        )
        # self.benchmark_analysis = ...

        # self.params = self.parameter_analysis.counts
        # self.params_sparse = self.parameter_analysis.counts_sparse
        # self.params_bits = self.parameter_analysis.bits
        # self.params_bits_quant = self.parameter_analysis.bits_quant

        # self.ops = self.operation_analysis.counts
        # self.ops_sparse = self.operation_analysis.counts_sparse
        # self.ops_bits = self.operation_analysis.bits
        # self.ops_bits_quant = self.operation_analysis.bits_quant

        # self.mem = self.memory_access_analysis.counts
        # self.mem_sparse = self.memory_access_analysis.counts_sparse
        # self.mem_bits = self.memory_access_analysis.bits
        # self.mem_bits_quant = self.memory_access_analysis.bits_quant

    def to_dict(self):
        ...

    def to_yaml(self):
        ...
