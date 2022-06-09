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
    ops_axes = param_axes.twinx()

    text_size = 8
    width = 0.3

    parameterized_prunable_nodes = [node for node in model_analysis.layers if node.parameterized_and_prunable]

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
        node_data["parameters_pos"].append(node_i - width / 2)
        node_data["ops_pos"].append(node_i + width / 2)
        node_data["sparse_parameters"].append(node.num_sparse_parameters)
        node_data["dense_parameters"].append(node.num_parameters - node.num_sparse_parameters)
        node_data["sparse_ops"].append(node.num_sparse_ops)
        node_data["dense_ops"].append(node.num_dense_ops)

    param_axes.bar(node_data["parameters_pos"], node_data["sparse_parameters"], width=width, color="deeppink", label="sparse parameters")
    param_axes.bar(node_data["parameters_pos"], node_data["dense_parameters"], bottom=node_data["sparse_parameters"], width=width, color="blue", label="dense parameters")

    ops_axes.bar(node_data["ops_pos"], node_data["sparse_ops"], width=width, color="purple", label="sparse operations")
    ops_axes.bar(node_data["ops_pos"], node_data["dense_ops"], bottom=node_data["sparse_ops"], width=width, color="darkblue", label="dense operations")

    param_axes.set_xticks(numpy.arange(len(parameterized_prunable_nodes)))
    param_axes.set_xticklabels(node_data["node_names"])
    param_axes.invert_xaxis()
    param_axes.set_ylabel('Number of parameters')
    param_axes.legend(loc="upper left")

    ops_axes.set_xticks(numpy.arange(len(parameterized_prunable_nodes)))
    ops_axes.invert_xaxis()
    ops_axes.set_ylabel('Number of floating-point/ integer operations')
    ops_axes.legend(loc="upper right")

    for node_i in range(len(parameterized_prunable_nodes)):
        total_parameters = node_data["dense_parameters"][node_i] + node_data["sparse_parameters"][node_i]
        if total_parameters > 0:
            parameter_sparsity = node_data["sparse_parameters"][node_i] / total_parameters
            param_axes.text(node_data["parameters_pos"][node_i], total_parameters, f"{parameter_sparsity:.0%}", color='black', fontweight='bold', size=text_size, verticalalignment="bottom", horizontalalignment="center")

        total_ops = node_data["dense_ops"][node_i] + node_data["sparse_ops"][node_i]
        if total_ops > 0:
            operation_sparsity = node_data["sparse_ops"][node_i] / total_ops
            ops_axes.text(node_data["ops_pos"][node_i], total_ops, f"{operation_sparsity:.0%}", color='black', fontweight='bold', size=text_size, verticalalignment="bottom", horizontalalignment="center")

    param_axes.set_title('Model parameter and operation sparsity')

    plt.setp(param_axes.get_xticklabels(), rotation=30, ha="right")

    plt.show()
