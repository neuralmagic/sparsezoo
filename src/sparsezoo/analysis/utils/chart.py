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
import matplotlib
import matplotlib.pyplot as plt

from sparsezoo.analysis import ModelAnalysis

def draw_param_sparsity_chart(model_analysis: ModelAnalysis, out_path: str):
    figure, param_axes = plt.subplots()
    ops_axes = param_axes.twiny()

    text_size = 8
    height = 0.3

    parameterized_prunable_nodes = [node for node in model_analysis.layers if node.parameterized_and_prunable]
    parameterized_prunable_nodes.reverse()

    node_data = {
        "node_names": [],
        "parameters_pos": [],
        "ops_pos": [],
        "sparse_parameters": [],
        "dense_parameters": [],
        "sparse_ops": [],
        "dense_ops": []
    }
    for node_i, node in enumerate(parameterized_prunable_nodes):
        node_data["node_names"].append(node.name)
        node_data["parameters_pos"].append(node_i + height / 2)
        node_data["ops_pos"].append(node_i - height / 2)
        node_data["sparse_parameters"].append(node.num_sparse_parameters)
        node_data["dense_parameters"].append(node.num_parameters - node.num_sparse_parameters)
        node_data["sparse_ops"].append(node.num_sparse_ops)
        node_data["dense_ops"].append(node.num_dense_ops)

    param_axes.barh(node_data["parameters_pos"], node_data["sparse_parameters"], height=height, color="deeppink", label="sparse parameters")
    param_axes.barh(node_data["parameters_pos"], node_data["dense_parameters"], left=node_data["sparse_parameters"], height=height, color="blue", label="dense parameters")

    ops_axes.barh(node_data["ops_pos"], node_data["sparse_ops"], height=height, color="purple", label="sparse operations")
    ops_axes.barh(node_data["ops_pos"], node_data["dense_ops"], left=node_data["sparse_ops"], height=height, color="darkblue", label="dense operations")

    param_axes.set_yticks(numpy.arange(len(parameterized_prunable_nodes)))
    param_axes.set_yticklabels(node_data["node_names"])
    param_axes.invert_yaxis()
    param_axes.set_xlabel('Number of parameters')
    param_axes.legend(loc="upper left")

    ops_axes.set_yticks(numpy.arange(len(parameterized_prunable_nodes)))
    ops_axes.invert_yaxis()
    ops_axes.set_xlabel('Number of floating-point/ integer operations')
    ops_axes.legend(loc="upper right")

    for node_i in range(len(parameterized_prunable_nodes)):
        parameter_sparsity = node_data["sparse_parameters"][node_i] / (node_data["dense_parameters"][node_i] + node_data["sparse_parameters"][node_i])
        operation_sparsity = node_data["sparse_ops"][node_i] / (node_data["dense_ops"][node_i] + node_data["sparse_ops"][node_i])
        param_axes.text(node_data["sparse_parameters"][node_i] + node_data["dense_parameters"][node_i] + 50, node_data["parameters_pos"][node_i], f"{parameter_sparsity:.0%}", color='black', fontweight='bold', size=text_size, verticalalignment="center")
        ops_axes.text(node_data["sparse_ops"][node_i] + node_data["dense_ops"][node_i] + 50, node_data["ops_pos"][node_i], f"{operation_sparsity:.0%}", color='black', fontweight='bold', size=text_size, verticalalignment="center")

    param_axes.set_title('Model parameter and operation sparsity')

    plt.setp(param_axes.get_yticklabels(), rotation=30)

    plt.show()
