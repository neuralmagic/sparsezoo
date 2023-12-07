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

  MODEL_PATH: can be path to the onnx file or sparsezoo stub

Options:
  --save TEXT         File path or directory to save the results as yaml to

##########
Usage:
    sparsezoo.analyze path/to/model.onnx         # runs model analysis
    sparsezoo.analyze stub --save analysis.yaml  # saves yaml as analysis.yaml
"""
import logging
from typing import Optional

import click
from sparsezoo.analytics import sparsezoo_analytics
from sparsezoo.analyze_v2 import analyze


__all__ = ["main"]


_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument(
    "path",
    type=str,
    required=True,
)
@click.option(
    "--save",
    default=None,
    type=str,
    help="File path or directory to save the results as yaml to Note: file will be "
    "overwritten if exists, if a directory path is passed results will be "
    "stored under the directory as `analysis.yaml`, if set to True the file "
    "will be saved under CWD as `analysis.yaml`",
)
@sparsezoo_analytics.send_event_decorator("cli__main")
def main(
    path: str,
    save: Optional[str],
):
    """
    Model analysis for ONNX models.
    Runs analysis for nodes with weights

    Usage:
    sparsezoo.analyze path/to/model.onnx
    sparsezoo.analyze stub
    """
    logging.basicConfig(level=logging.INFO)
    analysis = analyze(path)

    if save:
        _LOGGER.info(f"Writing results to {save}")
        with open(save, "w") as file:
            file.write(analysis.to_yaml())


if __name__ == "__main__":
    main()
