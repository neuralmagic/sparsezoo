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
#####################
Command help:
usage: sparsezoo.analysis [-h] [--outfile OUTFILE] model_path

SparseZoo Model Analysis utility for ONNX models

positional arguments:
  model_path         Path to a local ONNX model or SparseZoo model stub like z
                     oo:cv/classification/resnet_v1-50/pytorch/sparseml/imagen
                     et/pruned-moderate

optional arguments:
  -h, --help         show this help message and exit
  --outfile OUTFILE  If specified model analysis results will be written to
                     this file

#####################
Examples:
1) sparsezoo.analysis \
        zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate
2) sparsezoo.analysis ~/models/resnet50.onnx --outfile resnet50-analysis.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

from sparsezoo import Model
from sparsezoo.analysis import ModelAnalysis


__all__ = ["main"]

LOGGER = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="SparseZoo Model Analysis utility for ONNX models"
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to a local ONNX model or SparseZoo model stub like "
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned"
        "-moderate",
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="If specified model analysis results will be written to this file",
    )
    return parser.parse_args()


def main():
    """
    Driver function for the script
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if not isinstance(args.model_path, str):
        raise ValueError("model_path be a string")

    if args.model_path.startswith("zoo:"):
        LOGGER.info(f"Downloading files from SparseZoo: '{args.model_path}'")
        model = Model(args.model_path)
        model_path = Path(model.deployment.path) / "model.onnx"
    else:
        model_path = args.model_path

    analysis = ModelAnalysis.from_onnx(model_path)
    print(f"Model Analysis results for {args.model_path}")
    yaml.dump(analysis.dict(), stream=sys.stdout)

    if args.outfile:
        LOGGER.info(f"Writing results to {args.outfile}")
        analysis.yaml(args.outfile)


if __name__ == "__main__":
    main()
