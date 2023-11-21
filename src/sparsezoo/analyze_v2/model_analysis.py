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
import onnx

from sparsezoo.utils import ONNXGraph, extract_node_id, extract_node_shapes_and_dtypes

from .node_analysis import NodeAnalysis


class ModelAnalysis:
    def __init__(self, path):
        self.path = path
        self.nodes = {}  # key node_id

        self.param_counts = 0
        self.param_counts_sparse = 0
        self.param_bits = 0
        self.param_bits_quant = 0

        self.ops_counts = 0
        self.ops_counts_sparse = 0
        self.ops_bits = 0
        self.ops_bits_quant = 0

        self.mem_counts = 0
        self.mem_sparse = 0
        self.mem_bits = 0
        self.mem_bits_quant = 0

    def run(self, data: numpy.ndarray):

        model = onnx.load(self.path)
        model_graph = ONNXGraph(model)

        node_shapes, node_dtypes = extract_node_shapes_and_dtypes(model_graph.model)

        for node in model_graph.nodes:
            node_id = extract_node_id(node)
            node_shape = node_shapes.get(node_id)
            node_dtype = node_dtypes.get(node_id)
            self.nodes[node_id] = NodeAnalysis(
                model_graph, node, node_shape, node_dtype
            )

            self._update_summary_analysis_metadata(node_id)

        self.summary = None  # ModelSummary()
        self.parameters = None  # ModelParameter
        self.nodes = None  # ModelNode

        print(self.param_counts_sparse / self.param_counts)
        print(self.param_bits_quant / self.param_bits)

        print(self.ops_counts_sparse / self.ops_counts)
        print(self.ops_bits_quant / self.ops_bits)

        print(self.mem_sparse / self.mem_counts)
        print(self.mem_bits_quant / self.mem_bits)

    def _update_summary_analysis_metadata(self, node_id):

        # params
        self.param_counts += self.nodes[node_id].parameter_analysis.counts["single"][
            "counts"
        ]
        self.param_counts_sparse += self.nodes[node_id].parameter_analysis.counts[
            "single"
        ]["counts_sparse"]
        self.param_bits += self.nodes[node_id].parameter_analysis.bits["tensor"]["bits"]
        self.param_bits_quant += self.nodes[node_id].parameter_analysis.bits["tensor"][
            "bits_quant"
        ]

        # ops
        self.ops_counts += self.nodes[node_id].operation_analysis.counts["single"][
            "counts"
        ]
        self.ops_counts_sparse += self.nodes[node_id].operation_analysis.counts[
            "single"
        ]["counts_sparse"]
        self.ops_bits += self.nodes[node_id].operation_analysis.bits["tensor"]["bits"]
        self.ops_bits_quant += self.nodes[node_id].operation_analysis.bits["tensor"][
            "bits_quant"
        ]

        # breakpoint()
        self.mem_counts += self.nodes[node_id].memory_access_analysis.counts["single"][
            "counts"
        ]
        self.mem_sparse += self.nodes[node_id].memory_access_analysis.counts["single"][
            "counts_sparse"
        ]
        self.mem_bits += self.nodes[node_id].memory_access_analysis.bits["tensor"][
            "bits"
        ]
        self.mem_bits_quant += self.nodes[node_id].memory_access_analysis.bits[
            "tensor"
        ]["bits_quant"]

        # mem access


def analyze(
    path: str,
    data: numpy.ndarray,
):
    analysis = ModelAnalysis(path)
    analysis.run(data)
