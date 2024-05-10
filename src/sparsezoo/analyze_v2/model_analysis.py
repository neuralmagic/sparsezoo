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


from typing import Dict, Optional

import onnx
import yaml

from sparsezoo import Model
from sparsezoo.analyze_v2.node_analysis import NodeAnalysis
from sparsezoo.analyze_v2.schemas import ModelAnalysisSchema
from sparsezoo.analyze_v2.summary_analysis import SummaryAnalysis
from sparsezoo.model.utils import is_stub
from sparsezoo.utils import (
    ONNXGraph,
    extract_node_id,
    extract_node_shapes_and_dtypes,
    load_model,
)


class ModelAnalysis:
    """
    Given summary and node anlysis, obtain the overall analysis of the onnx model.

    From fine-grained detailed analysis (top):

    SparsityAnalysis:       num_counts, num_counts_sparse
    MemoryAccessAnalysis:   num_mem_access, num_mem_access_sparse
    QuantizationAnalysis:   num_bits, num_bits_quant

    ParameterAnalysis:      contains SparsityAnalysis, MemoryAccessAnalysis,
                             QuantizationAnalysis
    OperationAnalysis:      contains SparsityAnalysis, MemoryAccessAnalysis,
                             QuantizationAnalysis
    DistributionAnalysis:   contains SparsityAnalysis, MemoryAccessAnalysis,
                             QuantizationAnalysis

    NodeAnalysis:           contains ParameterAnalysis, OperationAnalysis,
                             DistributionAnalysis per node_id
    SummaryAnalysis         contains the sum of SparsityAnalysis, MemoryAccessAnalysis,
                             QuantizationAnalysis per grouping
    ModelAnalysis           contains NodeAnalysis, SummaryAnalysis
    """

    def __init__(
        self, summary_analysis: SummaryAnalysis, node_analyses: Dict[str, NodeAnalysis]
    ):
        self.summary_analysis = summary_analysis
        self.node_analyses = node_analyses

    def to_dict(self):
        summaries = self.summary_analysis.to_dict()
        nodes = {}
        for id, node in self.node_analyses.items():
            if node.to_dict() is not None:
                nodes[id] = node.to_dict()

        return ModelAnalysisSchema(
            summaries=summaries,
            nodes=nodes,
        ).dict()

    def calculate_sparsity_percentage(self, category: Dict):
        counts_sparse = category["counts_sparse"]
        counts = category["counts"]
        return (counts_sparse / counts) * 100 if counts != 0 else 0

    def calculate_quantized_percentage(self, tensor: Dict, counts_prefix: str):
        counts_quant = tensor[f"{counts_prefix}_quant"]
        counts = tensor[counts_prefix]
        return (counts_quant / counts) * 100 if counts != 0 else 0

    def __repr__(self):
        data = self.to_dict()
        summaries = data["summaries"]

        param_total = summaries["params"]["sparsity"]["single"]["counts"]
        param_sparsity = self.calculate_sparsity_percentage(
            summaries["params"]["sparsity"]["single"]
        )
        param_size = summaries["params"]["quantization"]["tensor"]["bits"]
        param_quantized = self.calculate_quantized_percentage(
            summaries["params"]["quantization"]["tensor"], "counts"
        )

        ops_total = summaries["ops"]["sparsity"]["single"]["counts"]
        ops_sparsity = self.calculate_sparsity_percentage(
            summaries["ops"]["sparsity"]["single"]
        )
        ops_size = summaries["ops"]["quantization"]["tensor"]["bits"]
        ops_quantized = self.calculate_quantized_percentage(
            summaries["ops"]["quantization"]["tensor"], "counts"
        )

        mem_access_total = summaries["mem_access"]["sparsity"]["single"]["counts"]
        mem_access_sparsity = self.calculate_sparsity_percentage(
            summaries["mem_access"]["sparsity"]["single"]
        )
        mem_access_size = summaries["mem_access"]["quantization"]["tensor"]["bits"]
        mem_access_quantized = self.calculate_quantized_percentage(
            summaries["mem_access"]["quantization"]["tensor"], "counts"
        )

        return (
            "Params:\n"
            f"\ttotal\t\t: {param_total}\n"
            f"\tsparsity%\t: {param_sparsity}\n"
            f"\tsize [bits]\t: {param_size}\n"
            f"\tquantized %\t: {param_quantized}\n"
            "Ops:\n"
            f"\ttotal\t\t: {ops_total}\n"
            f"\tsparsity%\t: {ops_sparsity}\n"
            f"\tsize [bits]\t: {ops_size}\n"
            f"\tquantized %\t: {ops_quantized}\n"
            "Memory Access:\n"
            f"\ttotal\t\t: {mem_access_total}\n"
            f"\tsparsity%\t: {mem_access_sparsity}\n"
            f"\tsize [bits]\t: {mem_access_size}\n"
            f"\tquantized %\t: {mem_access_quantized}\n"
        )

    def to_yaml(self):
        return yaml.dump(self.to_dict())


def analyze(path: str, download_path: Optional[str] = None) -> "ModelAnalysis":
    """
    Entry point to run the model analysis.

    Carries out analysis of the desired onnx model with respect
    to the weights in the nodes, if exists.

    :param path: .onnx path or stub
    """
    if path.endswith(".onnx"):
        onnx_model = load_model(path, load_external_data=False)
        onnx_model_path = path
    elif is_stub(path):
        model = Model(path, download_path)
        onnx_model_path = model.onnx_model.path
        onnx_model = onnx.load(onnx_model_path, load_external_data=False)
    else:
        raise ValueError(f"{path} is not a valid argument")

    # just need graph to get shape information; dont load external data
    node_shapes, _ = extract_node_shapes_and_dtypes(onnx_model, onnx_model_path)

    summary_analysis = SummaryAnalysis()
    node_analyses = {}

    # load external data for node analysis
    onnx_model = onnx.load(onnx_model_path)
    model_graph = ONNXGraph(onnx_model)

    for graph_order, node in enumerate(model_graph.nodes):
        node_id = extract_node_id(node)
        node_shape = node_shapes.get(node_id)

        analysis = NodeAnalysis(model_graph, node, node_shape, graph_order + 1)
        node_analyses[node_id] = analysis

        summary_analysis.aggregate_analysis_metrics_from_node_analysis(analysis)

    model_analysis = ModelAnalysis(
        summary_analysis=summary_analysis,
        node_analyses=node_analyses,
    )
    return model_analysis
