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
import copy
import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from sparsezoo import Model
from sparsezoo.analyze import ModelAnalysis


__all__ = ["main"]

from sparsezoo.analyze.cli import CONTEXT_SETTINGS, analyze_options


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

    model_file_path = _get_model_file_path(model_path=model_path)

    LOGGER.info("Starting Analysis ...")
    analysis = ModelAnalysis.from_onnx(model_file_path)
    LOGGER.info("Analysis complete, collating results...")

    summary = analysis.summary()

    summary["MODEL"] = model_path
    _display_summary_as_table(summary)

    if save:
        LOGGER.info(f"Writing results to {save}")
        analysis.yaml(file_path=save)


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


def _display_summary_as_table(summary):
    summary_copy = copy.copy(summary)
    print(f"MODEL: {summary_copy.pop('MODEL')}", end="\n\n")
    footer = summary_copy.pop("Summary")

    for section_name, section_dict in summary_copy.items():
        print(f"{section_name.upper()}:")
        print(pd.DataFrame(section_dict).T.to_string(), end="\n\n")

    print("SUMMARY:")
    for footer_key, footer_value in footer.items():
        print(f"{footer_key}: {footer_value}")


if __name__ == "__main__":
    main()
