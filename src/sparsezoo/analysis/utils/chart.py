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

from typing import Optional, Tuple

import numpy

import matplotlib.pyplot as plt
from sparsezoo.analysis import ModelAnalysis, NodeAnalysis


__all__ = [
    "draw_sparsity_by_layer_chart",
    "draw_parameter_chart",
    "draw_operation_chart",
    "draw_parameter_operation_combined_chart",
]


def draw_sparsity_by_layer_chart(
    model_analysis: ModelAnalysis,
    out_path: Optional[str] = None,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (25, 15),
) -> None:
    """
    Draws a figure that shows the precent sparsity of each layer in the model.
    Note that although this graph shows percent parameter sparsity, perecent
    operation sparsity is nearly the same for all prunable node op types.

    :param model_analysis: analysis of model whose chart is being drawn
    :param out_path: optional file path to save chart to
    :param model_name: name of model being analyzed, used in chart title
    :param figsize: keyword argument to pass to matplotlib figure
    :return: None
    """
    figure, axes = plt.subplots(figsize=figsize)

    # Set title
    figure.suptitle(f"{model_name} Sparsity per Layer")

    # Ingest node data
    parameterized_prunable_nodes = [
        node for node in model_analysis.nodes if node.parameterized_prunable
    ]
    node_data = {
        "names": [],
        "sparsities": [],
    }
    for node_i, node in enumerate(parameterized_prunable_nodes):
        node_data["names"].append(_get_node_name(node))
        node_data["sparsities"].append(
            node.parameter_summary.block_structure["single"].sparsity * 100
        )

    # Draw chart data
    axes.bar(
        numpy.arange(len(node_data["names"])),
        [100 - sparsity for sparsity in node_data["sparsities"]],
        bottom=node_data["sparsities"],
        color="blue",
        label="non-zero parameters",
    )
    axes.bar(
        numpy.arange(len(node_data["names"])),
        node_data["sparsities"],
        color="deeppink",
        label="zero parameters",
    )

    # Draw labels
    axes.set_xticks(numpy.arange(len(node_data["names"])))
    axes.set_xticklabels(node_data["names"])
    plt.setp(axes.get_xticklabels(), rotation=35, ha="right")
    axes.set_ylabel("Sparsity (%)")
    axes.legend(loc="upper left")

    # Reduce margins to zero
    axes.margins(0)

    # Save to file
    if out_path:
        plt.savefig(out_path)


def draw_parameter_chart(
    model_analysis: ModelAnalysis,
    out_path: Optional[str] = None,
    model_name: str = "Model",
    text_size: float = 8.0,
    bar_width: float = 0.3,
    figsize: Tuple[int, int] = (25, 15),
) -> None:
    """
    Draws a figure that shows sparse and dense parameter counts per layer

    :param model_analysis: analysis of model whose chart is being drawn
    :param out_path: optional file path to save chart to
    :param model_name: name of model being analyzed, used in chart title
    :param text_size: size of text to use for percent sparse labels
    :param bar_width: width of bars in chart
    :param figsize: keyword argument to pass to matplotlib figure
    :return: None
    """
    figure, param_axes = plt.subplots(figsize=figsize)

    # Set title
    figure.suptitle(f"{model_name} Number of Parameters per Layer")

    # Ingest node data
    parameterized_prunable_nodes = [
        node for node in model_analysis.nodes if node.parameterized_prunable
    ]
    node_data = {"names": [], "zero_parameters": [], "non_zero_parameters": []}
    for node_i, node in enumerate(parameterized_prunable_nodes):
        node_data["names"].append(_get_node_name(node))
        node_data["zero_parameters"].append(
            node.parameter_summary.block_structure["single"].zero
        )
        node_data["non_zero_parameters"].append(
            node.parameter_summary.block_structure["single"].non_zero
        )

    # Draw parameters bars
    param_axes.bar(
        numpy.arange(len(node_data["names"])),
        node_data["non_zero_parameters"],
        bottom=node_data["zero_parameters"],
        width=bar_width,
        color="blue",
        label="non-zero parameters",
    )
    param_axes.bar(
        numpy.arange(len(node_data["names"])),
        node_data["zero_parameters"],
        width=bar_width,
        color="deeppink",
        label="zero parameters",
    )

    param_axes.set_xticks(numpy.arange(len(parameterized_prunable_nodes)))
    param_axes.set_ylabel("Number of Parameters")
    param_axes.legend(loc="upper left")

    # Draw percent sparse text
    for node_i in range(len(parameterized_prunable_nodes)):
        total_parameters = (
            node_data["non_zero_parameters"][node_i]
            + node_data["zero_parameters"][node_i]
        )
        if total_parameters > 0:
            parameter_sparsity = node_data["zero_parameters"][node_i] / total_parameters
            param_axes.text(
                node_i,
                total_parameters,
                f"{parameter_sparsity:.0%}",
                color="black",
                fontweight="bold",
                size=text_size,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

    # Draw x-axis labels
    param_axes.set_xticklabels(node_data["names"])
    plt.setp(param_axes.get_xticklabels(), rotation=35, ha="right")

    # Save to file
    if out_path:
        plt.savefig(out_path)


def draw_operation_chart(
    model_analysis: ModelAnalysis,
    out_path: Optional[str] = None,
    model_name: str = "Model",
    text_size: float = 8.0,
    bar_width: float = 0.3,
    figsize: Tuple[int, int] = (25, 15),
) -> None:
    """
    Draws a figure that shows sparse and dense operation counts per layer

    :param model_analysis: analysis of model whose chart is being drawn
    :param out_path: optional file path to save chart to
    :param model_name: name of model being analyzed, used in chart title
    :param text_size: size of text to use for percent sparse labels
    :param bar_width: width of bars in chart
    :param figsize: keyword argument to pass to matplotlib figure
    :return: None
    """
    figure, ops_axes = plt.subplots(figsize=figsize)

    # Set title
    figure.suptitle(
        f"{model_name} Number of Floating-point / Integer Operations per Layer"
    )

    # Ingest node data
    parameterized_prunable_nodes = [
        node for node in model_analysis.nodes if node.parameterized_prunable
    ]
    node_data = {"names": [], "sparse_ops": [], "dense_ops": []}
    for node_i, node in enumerate(parameterized_prunable_nodes):
        node_data["names"].append(_get_node_name(node))
        node_data["sparse_ops"].append(
            node.operation_summary.ops.block_structure["single"].sparse
        )
        node_data["dense_ops"].append(
            node.operation_summary.ops.block_structure["single"].dense
        )

    # Draw operations bars
    ops_axes.bar(
        range(len(node_data["names"])),
        node_data["dense_ops"],
        bottom=node_data["sparse_ops"],
        width=bar_width,
        color="darkblue",
        label="dense operations",
    )
    ops_axes.bar(
        range(len(node_data["names"])),
        node_data["sparse_ops"],
        width=bar_width,
        color="purple",
        label="sparse operations",
    )

    ops_axes.set_xticks(numpy.arange(len(parameterized_prunable_nodes)))
    ops_axes.set_ylabel("Number of Floating-point/ Integer Operations")
    ops_axes.legend(loc="upper left")

    # Draw percent sparse text
    for node_i in range(len(parameterized_prunable_nodes)):
        total_ops = node_data["dense_ops"][node_i] + node_data["sparse_ops"][node_i]
        if total_ops > 0:
            operation_sparsity = node_data["sparse_ops"][node_i] / total_ops
            ops_axes.text(
                node_i,
                total_ops,
                f"{operation_sparsity:.0%}",
                color="black",
                fontweight="bold",
                size=text_size,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

    # Draw x-axis labels
    ops_axes.set_xticklabels(node_data["names"])
    plt.setp(ops_axes.get_xticklabels(), rotation=35, ha="right")

    # Save to file
    if out_path:
        plt.savefig(out_path)


def draw_parameter_operation_combined_chart(
    model_analysis: ModelAnalysis,
    out_path: Optional[str] = None,
    model_name: str = "Model",
    text_size: float = 8.0,
    bar_width: float = 0.3,
    figsize: Tuple[int, int] = (25, 15),
) -> None:
    """
    Draws a figure that shows sparse and dense parameter and operation counts per layer

    :param model_analysis: analysis of model whose chart is being drawn
    :param out_path: optional file path to save chart to
    :param model_name: name of model being analyzed, used in chart title
    :param text_size: size of text to use for percent sparse labels
    :param bar_width: width of bars in chart
    :param figsize: keyword argument to pass to matplotlib figure
    :return: None
    """
    figure, param_axes = plt.subplots(figsize=figsize)
    ops_axes = param_axes.twinx()

    # Set title
    figure.suptitle(f"{model_name} Number of Parameters and Operations per Layer")

    # Ingest node data
    parameterized_prunable_nodes = [
        node for node in model_analysis.nodes if node.parameterized_prunable
    ]
    node_data = {
        "names": [],
        "parameters_pos": [],
        "ops_pos": [],
        "zero_parameters": [],
        "non_zero_parameters": [],
        "sparse_ops": [],
        "dense_ops": [],
    }
    for node_i, node in enumerate(parameterized_prunable_nodes):
        node_data["names"].append(_get_node_name(node))
        node_data["parameters_pos"].append(node_i - bar_width / 2)
        node_data["ops_pos"].append(node_i + bar_width / 2)
        node_data["zero_parameters"].append(
            node.parameter_summary.block_structure["single"].zero
        )
        node_data["non_zero_parameters"].append(
            node.parameter_summary.block_structure["single"].non_zero
        )
        node_data["sparse_ops"].append(
            node.operation_summary.ops.block_structure["single"].sparse
        )
        node_data["dense_ops"].append(
            node.operation_summary.ops.block_structure["single"].dense
        )

    # Draw parameters bars
    param_axes.bar(
        node_data["parameters_pos"],
        node_data["non_zero_parameters"],
        bottom=node_data["zero_parameters"],
        width=bar_width,
        color="blue",
        label="non-zero parameters",
    )
    param_axes.bar(
        node_data["parameters_pos"],
        node_data["zero_parameters"],
        width=bar_width,
        color="deeppink",
        label="zero parameters",
    )

    param_axes.set_xticks(numpy.arange(len(parameterized_prunable_nodes)))
    param_axes.set_ylabel("Number of Parameters")
    param_axes.legend(loc="upper left")

    # Draw operations bars
    ops_axes.bar(
        node_data["ops_pos"],
        node_data["dense_ops"],
        bottom=node_data["sparse_ops"],
        width=bar_width,
        color="darkblue",
        label="dense operations",
    )
    ops_axes.bar(
        node_data["ops_pos"],
        node_data["sparse_ops"],
        width=bar_width,
        color="purple",
        label="sparse operations",
    )

    ops_axes.set_xticks(numpy.arange(len(parameterized_prunable_nodes)))
    ops_axes.set_ylabel("Number of Floating-point/ Integer Operations")
    ops_axes.legend(loc="upper right")

    # Draw percent sparse text
    for node_i in range(len(parameterized_prunable_nodes)):
        total_parameters = (
            node_data["non_zero_parameters"][node_i]
            + node_data["zero_parameters"][node_i]
        )
        if total_parameters > 0:
            parameter_sparsity = node_data["zero_parameters"][node_i] / total_parameters
            param_axes.text(
                node_data["parameters_pos"][node_i],
                total_parameters,
                f"{parameter_sparsity:.0%}",
                color="black",
                fontweight="bold",
                size=text_size,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

        total_ops = node_data["dense_ops"][node_i] + node_data["sparse_ops"][node_i]
        if total_ops > 0:
            operation_sparsity = node_data["sparse_ops"][node_i] / total_ops
            ops_axes.text(
                node_data["ops_pos"][node_i],
                total_ops,
                f"{operation_sparsity:.0%}",
                color="black",
                fontweight="bold",
                size=text_size,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

    # Draw x-axis labels
    param_axes.set_xticklabels(node_data["names"])
    plt.setp(param_axes.get_xticklabels(), rotation=35, ha="right")

    # Save to file
    if out_path:
        plt.savefig(out_path)


def _get_node_name(node_analysis: NodeAnalysis) -> str:
    """
    Pick an intuitive name for the node based on its name and weight name

    :param node: analysis of node whose name is being picked
    :return: an intutive name for this node
    """
    node_weight_analysis = next(
        (
            parameter
            for parameter in node_analysis.parameters
            if parameter.alias == "weight"
        ),
        None,
    )
    if (
        node_analysis.op_type == "Gather"
        and node_weight_analysis is not None
        and ".embeddings." in node_weight_analysis.name
    ):
        name = node_weight_analysis.name
        name = name.split(".embeddings.")[1]
        name = name.replace("weight", "")
        return name
    if node_analysis.quantized_node:
        return node_analysis.name
    if (
        node_weight_analysis is not None
        and node_weight_analysis.name
        and ".weight" in node_weight_analysis.name
    ):
        return node_weight_analysis.name.replace(".weight", "")
    else:
        return node_analysis.name
