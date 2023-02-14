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

"""
Usage: sparsezoo.analyze [OPTIONS] MODEL_PATH

  Model analysis for ONNX models.

  MODEL_PATH: can be a SparseZoo stub, or local path to a deployment-directory
  or ONNX model

  Examples:

  - Run model analysis on resnet

      sparsezoo.analyze ~/models/resnet50.onnx

Options:
  --save FILE  Path to a yaml file to write results to, note: file will be
               overwritten if exists
  --help       Show this message and exit.

##########
Examples:
1) Model Analysis on Resnet50
    sparsezoo.analyze \
    "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none"

2) Model Analysis on local file + saving results
    sparsezoo.analyze ~/models/resnet50.onnx \
    --save resnet50-analysis.yaml
"""
import logging
import pprint as pp
from collections import defaultdict
from pathlib import Path
from typing import Optional

import click
from sparsezoo import Model
from sparsezoo.analysis import ModelAnalysis
from sparsezoo.analysis.utils.models import ZeroNonZeroParams


__all__ = ["main"]


pp = pp.PrettyPrinter(indent=4, width=80, compact=True, sort_dicts=False)
LOGGER = logging.getLogger()


@click.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"),
        show_default=True,
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument(
    "model_path",
    type=str,
    required=True,
)
@click.option(
    "--save",
    default=None,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    help="Path to a yaml file to write results to, note: file will be "
    "overwritten if exists",
)
def main(model_path: str, save: Optional[str]):
    """
    Model analysis for ONNX models.

    MODEL_PATH: can be a SparseZoo stub, or local path to a
    deployment-directory or ONNX model

    Examples:

    - Run model analysis on resnet

        sparsezoo.analyze ~/models/resnet50.onnx
    """
    logging.basicConfig(level=logging.INFO)
    model_file_path = _get_model_file_path(model_path=model_path)

    LOGGER.info("Starting Analysis ...")
    analysis = ModelAnalysis.from_onnx(model_file_path)
    LOGGER.info("Analysis complete, collating results...")

    summary = _get_analysis_summary(analysis=analysis)
    summary["model_path"] = model_path
    pp.pprint(summary)

    if save:
        LOGGER.info(f"Writing results to {save}")
        analysis.yaml(file_path=save)


def _get_analysis_summary(analysis):
    def _get_count_dict(param_summary: "ZeroNonZeroParams"):
        return dict(
            total=param_summary.zero + param_summary.non_zero,
            pruned=param_summary.zero,
            pruned_percentage=f"{param_summary.sparsity * 100:.2f} %",
        )

    pruned_percentage = (
        analysis.parameter_summary.pruned * 100.0 / analysis.parameter_summary.total
    )
    parameter_count_summary = dict(
        total=analysis.parameter_summary.total,
        pruned=analysis.parameter_summary.pruned,
        pruned_percentage=f"{pruned_percentage:.2f} %",
    )

    # single, block, ...
    for sparsity_type, params in analysis.parameter_summary.block_structure.items():
        parameter_count_summary[sparsity_type] = _get_count_dict(param_summary=params)

    # fp32, int8, ...
    for param_type, params in analysis.parameter_summary.precision.items():
        parameter_count_summary[param_type] = _get_count_dict(param_summary=params)

    parameterized_operations_summary = analysis.parameterized.dict()
    non_parameterized_operations_summary = analysis.non_parameterized.dict()

    # basic parameterized and non parameterized ops summary
    for op_dict in (
        parameterized_operations_summary,
        non_parameterized_operations_summary,
    ):
        for param_type in ["pruned", "prunable", "quantized"]:
            percentage = (
                op_dict[param_type] * 100.0 / op_dict["total"]
                if op_dict["total"]
                else 0
            )
            if percentage:
                op_dict[f"{param_type}_percentage"] = f"{percentage:.2f} %"

    sparse_param_count = defaultdict(int)
    total_param_count = defaultdict(int)
    parameterized_op_types = set()
    non_parameterized_op_types = set()

    # sparse and total param count per operation type
    for node in analysis.nodes:
        sparse_param_count[node.op_type] += node.parameter_summary.pruned
        total_param_count[node.op_type] += node.parameter_summary.total
        if node.parameterized_prunable:
            parameterized_op_types.add(node.op_type)
        else:
            non_parameterized_op_types.add(node.op_type)

    # sparsity and counts for parameterized ops
    for node_type in parameterized_op_types:
        sparsity = (
            sparse_param_count[node_type] * 100.0 / total_param_count[node_type]
            if total_param_count[node_type]
            else 0
        )
        if sparsity:
            parameterized_operations_summary[
                f"{node_type}_sparsity"
            ] = f"{sparsity:.2f} %"
        if analysis.node_counts[node_type]:
            parameterized_operations_summary[
                f"{node_type}_count"
            ] = analysis.node_counts[node_type]

    # sparsity and counts for non-parameterized ops
    for node_type in non_parameterized_op_types:
        sparsity = (
            sparse_param_count[node_type] * 1.0 / total_param_count[node_type]
            if total_param_count[node_type]
            else 0
        )
        if sparsity:
            non_parameterized_operations_summary[
                f"{node_type}_sparsity"
            ] = f"{sparsity:.2f} %"
        if analysis.node_counts[node_type]:
            non_parameterized_operations_summary[
                f"{node_type}_count"
            ] = analysis.node_counts[node_type]

    return dict(
        number_of_parameters=parameter_count_summary,
        parameterized_operations=parameterized_operations_summary,
        non_parameterized_operations=non_parameterized_operations_summary,
    )


def _get_model_file_path(model_path: str):
    if model_path.startswith("zoo:"):
        LOGGER.info(f"Downloading files from SparseZoo: '{model_path}'")
        model = Model(model_path)
        model_path = Path(model.deployment.get_file("model.onnx").path)
    elif Path(model_path).is_file():
        model_path = model_path
    else:
        model_path = Path(model_path) / "model.onnx"
    return model_path


if __name__ == "__main__":
    main()
