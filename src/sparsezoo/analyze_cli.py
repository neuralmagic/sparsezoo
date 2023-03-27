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
  --compare TEXT      A multi-arg comma separated list of model paths to
                      compare against; can accept local path to an onnx model,
                      a path to deployment folder containing a onnx model, a
                      SparseZoo stub, or a previously run analysis yaml file.
                      No comparision is run if omitted
  --save TEXT         File path or directory to save the results as yaml to
                      Note: file will be overwritten if exists, if a directory
                      path is passed results will be stored under the
                      directory as `analysis.yaml`, if set to True the file
                      will be saved under CWD as `analysis.yaml`
  --save-graphs TEXT  Directory to save the generated graphs to. if not
                      specified (default) no files will be saved. If set or
                      boolean string, saves the graphs in the CWD under
                      `analysis-graphs`; If set as directory path, graphs are
                      saved in the specified directory
  --by-types TEXT     A flag to enable analysis results by operator type. If
                      set or boolean string, generates and records the results
                      by operator type
  --by-layer TEXT     A flag to enable analysis results by layer type. If set
                      or boolean string, generates and records the results
                      across all layers
  --help              Show this message and exit.
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
from typing import Optional

import click
from sparsezoo.analyze import ModelAnalysis
from sparsezoo.analyze.cli import CONTEXT_SETTINGS, analyze_options


__all__ = ["main"]


LOGGER = logging.getLogger()


@click.command(context_settings=CONTEXT_SETTINGS)
@analyze_options
def main(model_path: str, save: Optional[str], **kwargs):
    """
    Model analysis for ONNX models.

    MODEL_PATH: can be a SparseZoo stub, or local path to a
    deployment-directory or ONNX model

    Examples:

    - Run model analysis on resnet

        sparsezoo.analyze ~/models/resnet50.onnx
    """
    logging.basicConfig(level=logging.INFO)

    for unimplemented_feat in ("compare", "by_layer", "by_types", "save_graphs"):
        if kwargs.get(unimplemented_feat):
            raise NotImplementedError(
                f"--{unimplemented_feat} has not been implemented yet"
            )

    LOGGER.info("Starting Analysis ...")
    analysis = ModelAnalysis.create(model_path)
    LOGGER.info("Analysis complete, collating results...")

    print(f"MODEL: {model_path}", end="\n\n")
    analysis.pretty_print_summary()

    if save:
        LOGGER.info(f"Writing results to {save}")
        analysis.yaml(file_path=save)


if __name__ == "__main__":
    main()
