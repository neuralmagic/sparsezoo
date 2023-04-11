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
File containing click CLI options and helpers for analyze api
"""

import functools

import click


CONTEXT_SETTINGS = dict(
    token_normalize_func=lambda x: x.replace("-", "_"),
    show_default=True,
    ignore_unknown_options=True,
    allow_extra_args=True,
)

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
SUPPORTED_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]


def analyze_options(command: click.Command):
    """
    A decorator that takes in a click command and adds analyze api options
    to it, this method is meant to be a single source of truth across all
    analyze api(s). This decorator can be directly imported and used on
    top of another click command.

    :param command: A click callable command
    :return: The same click callable but with analyze api options attached
    """

    @click.argument(
        "model_path",
        type=str,
        required=True,
    )
    @click.option(
        "--compare",
        default=None,
        type=str,
        help="A multi-arg comma separated list of model paths to compare "
        "against; can accept local path to an onnx model, a path to "
        "deployment folder containing a onnx model, a SparseZoo stub, or a "
        "previously run analysis yaml file. No comparision is run if omitted",
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
    @click.option(
        "--save-graphs",
        default=None,
        type=str,
        help="Directory to save the generated graphs to. if not specified "
        "(default) no files will be saved. If set or boolean string, "
        "saves the graphs in the CWD under `analysis-graphs`; If set as"
        " directory path, graphs are saved in the specified directory",
    )
    @click.option(
        "--by-types",
        default=None,
        type=str,
        help="A flag to enable analysis results by operator type. If set or "
        "boolean string, generates and records the results by operator type",
    )
    @click.option(
        "--by-layers",
        default=None,
        type=str,
        help="A flag to enable analysis results by layer type. If set or "
        "boolean string, generates and records the results across all layers",
    )
    @functools.wraps(command)
    def wrap_common_options(*args, **kwargs):
        """
        Wrapper that adds analyze options to command
        """
        return command(*args, **kwargs)

    return wrap_common_options


def analyze_performance_options(command: click.Command):
    """
    A decorator that takes in a click command and adds analyze api performance
    analysis options to it. This decorator can be directly imported and
    used on top of another click command.

    :param command: A click callable command
    :return: The same click callable but with analyze api options attached
    """

    @click.option(
        "--batch-size-latency",
        default=1,
        type=int,
        help="The batch size to run latency benchmarks at",
    )
    @click.option(
        "--impose",
        default=None,
        type=str,
        help="The sparsification options to impose on the model and run as a "
        "comparison; this is a multi arg list of any of the following, "
        "`sparse` or `pruned` to set the sparsity for all prunable layers to 85%,"
        " #.## a float [0,1] to set the sparsity for all prunable layers to, "
        "`quant` or `quantized` to enable quantization with int8 activations "
        "and weights for all quantizable layers, {precision} to enable "
        "quantization with {precision: int16, int8, int4, int2} activations and "
        "weights for all quantizable layers w_{precision} to enable quantization"
        " with {precision: int16, int8, int4, int2} weights for all quantizable "
        "layers a_{precision} to enable quantization with {precision: int16, "
        "int8, int4, int2} activations for all quantizable layers",
    )
    @click.option(
        "--batch-size-throughput",
        default=1,
        type=int,
        help="The batch size to run throughput benchmarks at",
    )
    @click.option(
        "--benchmark-engine",
        default=DEEPSPARSE_ENGINE,
        type=click.Choice(SUPPORTED_ENGINES, case_sensitive=False),
        help="The engine to run the benchmarks through",
    )
    @functools.wraps(command)
    def wrap_with_performance_options(*args, **kwargs):
        """
        Wrapper that adds analyze performance options to command
        """
        return command(*args, **kwargs)

    return wrap_with_performance_options
