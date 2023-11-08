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

from typing import Optional, Dict

import onnx

from sparsezoo.analysis.utils import get_distribution_analysis, get_operation_analysis, get_parameter_analysis
from sparsezoo.utils import (
    NodeDataType,
    NodeShape,
    ONNXGraph,
    extract_node_id,
    extract_node_shapes_and_dtypes,
    get_layer_and_op_counts,
    get_node_bias,
    get_node_bias_name,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_paramters,
    get_node_weight,
    get_node_weight_name,
    get_ops_dict,
    get_zero_point,
    # has_paramters,
    is_four_block_sparse_layer,
    is_parameterized_prunable_layer,
    is_quantized_layer,
    is_sparse_layer,
)


class ModelAnalysis:
    def __init__(self, path):
        self.path = path
        # self.operation_analyses: Dict[str, OperationAnalysis]

    def run(self, data: Optional[str] = None, detailed: bool = False):
        onnx_model = onnx.load(self.path)
        onnx_graph = ONNXGraph(onnx_model)

        node_shapes, node_dtypes = extract_node_shapes_and_dtypes(onnx_graph.model)

        prev = None
        for node in onnx_graph.nodes:
            ## check where this node belongs to
            node_id = extract_node_id(node)
            node_shape = node_shapes.get(node_id)
            node_dtype = node_dtypes.get(node_id)

            operation_analysis = get_operation_analysis(
                onnx_graph, node, node_shape, node_dtype
            )
            
            parameter_analysis = get_parameter_analysis(
                onnx_graph, node, node_shape, node_dtype
            )

        #     sparsify_analysis = get_sparsity_analysis(node, onnx_graph)
        #     quantization_analysis = get_quantization_analysis(node, onnx_graph)

        #     distribution_analysis = get_distribution_analysis(node, onnx_graph, node_shape, node_dtype)

        #     benchmark_analysis = get_benchmark_analysis(node, onnx_graph)

        #     node_analysis = get_node_analysis(node, onnx_graph)

        #     parameters = get_node_paramters(node, onnx_graph)
        #     if parameters > 0:
        #         param_analysis = get_param_analysis(node, onnx_graph)
        #     else:
        #         op_analysis = get_op_analysis(node, onnx_graph)
        #         memory_access_analysis = get_memory_access_analysis(node, onnx_graph)

        #     if prev.op_type in list(
        #         ONNXActivationFunctions.__members__.values().lower()
        #     ):
        #         ...

        # breakpoint()
        # summary_analysis = get_summary_analysis(...)


def analyze(
    model: str,
    data: Optional[str] = None,
    detailed: bool = False,
) -> bool:
    """
    Simple func to run model analysis
    :param model: Stub, .onnx path or a folder containing it
    :param data: Variables that represent the shape of the input data
    :param detailed:
    """
    model_analysis = ModelAnalysis(model)
    analysis = model_analysis.run(
        data=data,
        detailed=detailed,
    )
    return analysis.to_dict()
