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

import yaml

from sparsezoo.analyze_v2.model_validator import NodeAnalysisModel

from .memory_access_analysis import MemoryAccessAnalysis
from .operation_analysis import OperationAnalysis
from .parameter_analysis import ParameterAnalysis


class NodeAnalysis:
    def __init__(
        self, model_graph, node, node_shape, graph_order: Optional[int] = None
    ):
        self.node = node
        self.model_graph = model_graph
        self.node_shape = node_shape
        self.graph_order = graph_order
        self.parameter_analysis = ParameterAnalysis(model_graph, node)
        self.operation_analysis = OperationAnalysis(model_graph, node, node_shape)
        self.memory_access_analysis = MemoryAccessAnalysis(
            model_graph, node, node_shape
        )

        self.params_counts, self.params_counts_sparse = 0, 0
        self.params_bits, self.params_bits_quant = 0, 0

        self.ops_counts, self.ops_counts_sparse = 0, 0
        self.ops_bits, self.ops_bits_quant = 0, 0

        self.mem_counts, self.mem_counts_sparse = 0, 0
        self.mem_bits, self.mem_bits_quant = 0, 0

        self._update_metadata()

    def to_dict(self):
        return NodeAnalysisModel(
            name=self.node.name,
            op_type=self.node.op_type,
            graph_order=self.graph_order,
            input=self.node.input,
            output=self.node.output,
            params=self.parameter_analysis.to_dict(),
            ops=self.operation_analysis.to_dict(),
            mem_access=self.memory_access_analysis.to_dict(),
        ).dict()

    def to_yaml(self):
        return yaml.dump(self.to_dict())

    def _update_metadata(self):
        self.params_counts = self.parameter_analysis.counts["counts"]
        self.params_counts_sparse = self.parameter_analysis.counts["counts_sparse"]
        self.params_bits = self.parameter_analysis.bits["bits"]
        self.params_bits_quant = self.parameter_analysis.bits["bits_quant"]

        self.ops_counts = self.operation_analysis.counts["counts"]
        self.ops_counts_sparse = self.operation_analysis.counts["counts_sparse"]
        self.ops_bits = self.operation_analysis.bits["bits"]
        self.ops_bits_quant = self.operation_analysis.bits["bits_quant"]

        self.mem_counts = self.memory_access_analysis.counts["counts"]
        self.mem_counts_sparse = self.memory_access_analysis.counts["counts_sparse"]
        self.mem_bits = self.memory_access_analysis.bits["bits"]
        self.mem_bits_quant = self.memory_access_analysis.bits["bits_quant"]
