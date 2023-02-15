#!/usr/bin/env python

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

# flake8:noqa
"""
Usage: sparsezoo.deployment_package [OPTIONS] [DIRECTORY]

  Utility to fetch a deployment directory for a task based on a optimizing-
  metric

  Example for using sparsezoo.deployment_package:

       1) `sparsezoo.deployment_package --task image_classification -m accuracy`

       2) `sparsezoo.deployment_package --task ic --optimizing_metric accuracy
       --optimizing_metric compression --target VNNI`

Options:
  --version                       Show the version and exit.
  --task [ic|image-classification|image_classification|classification|od|
  object-detection|object_detection|detection|segmentation|qa|question-answering|
  question_answering|text-classification|text_classification|glue|sentiment|
  sentiment_analysis|sentiment-analysis|token-classification|token_classification|
  ner|named-entity-recognition|named_entity_recognition]
                                  The task to find model for, must be
                                  specified if `--dataset` not provided
  --dataset [imagenette|imagenet|coco|squad|mnli|qqp|sst2|conll2003]
                                  The public dataset used to train this model,
                                  must be specified if `--task` not provided
  --optimizing-metric, --optimizing_metric TEXT
                                  The criterion to search model for, multiple
                                  metrics can be specified as comma
                                  separated values, supported metrics are
                                  ['accuracy', 'f1', 'recall', 'mAP',
                                  'latency', 'throughput', 'compression',
                                  'file_size', 'memory_usage']  [default:
                                  accuracy]
  --target [VNNI|DEFAULT]         Deployment target scenario (ie 'VNNI' for
                                  VNNI capable CPUs)  [default: DEFAULT]
  --help                          Show this message and exit.  [default:
                                  False]
##########
Examples:
    1) Fetch the smallest Image Classification Model trained on imagenette
        sparsezoo.deployment_package --dataset imagenette --optimizing_metric compression
    2) Fetch the most accurate Image Classification  Model trained on imagenette
        sparsezoo.deployment_package --dataset imagenette --optimizing_metric accuracy
    3) Fetch the most performant Question Answering model trained on squad
        sparsezoo.deployment_package --task qa --dataset squad --optimizing_metric latency
    4) Fetch the smallest most performant Question Answering model trained on squad
        sparsezoo.deployment_package --task qa --dataset squad \
            --optimizing_metric "compression, accuracy"
"""
import logging
from typing import Any, Mapping, Optional

import click
from sparsezoo import Model, deployment_package
from sparsezoo.deployment_package.docker.helpers import DEPLOYMENT_DOCKER_PATH
from sparsezoo.utils import (
    DATASETS,
    DEFAULT_DEPLOYMENT_SCENARIO,
    DEFAULT_OPTIMIZING_METRIC,
    DEPLOYMENT_SCENARIOS,
    METRICS,
    TASKS_WITH_ALIASES,
)
from sparsezoo.version import __version__


_LOGGER = logging.getLogger(__name__)


def _csv_callback(ctx, self, value):
    """
    A click callback function to parse a comma separated string with metrics
    into a list
    """
    current_metrics = []
    for metric in value.split(","):
        metric_ = metric.lower().strip()
        if metric_ not in METRICS:
            raise ValueError(f"Specified metric {metric_} is not supported")
        current_metrics.append(metric_)
    return current_metrics


def _download_deployment_directory(stub: str, destination: Optional[str] = None) -> str:
    model = Model(stub)
    model.deployment.download(destination_path=destination)
    return model.deployment.path


def _get_template(results: Mapping[str, Any], destination: Optional[str] = None):
    stub = results.get("stub")

    if not stub:
        return "No relevant models found for specified metrics"

    stub_info = f"""
        Relevant Stub: {stub}
    """
    metrics = results.get("metrics")
    metrics_info = (
        f"""
        Model Metrics: {metrics}
        """
        if metrics
        else ""
    )
    dockerfile = DEPLOYMENT_DOCKER_PATH
    dockerfile_directory = DEPLOYMENT_DOCKER_PATH.parent
    deployment_path = _download_deployment_directory(stub, destination=destination)
    download_instructions = f"""
        Use the dockerfile in {dockerfile} to build deepsparse
        image and run the `deepsparse.server` container. 

        Run the following command inside `{dockerfile_directory}`
        directory (Note: replace <TASK-NAME> with appropriate task):

        ```bash        
        docker build -t deepsparse_docker . && docker run -it \\
        -v {deployment_path}:/home/deployment  deepsparse_docker \\
         deepsparse.server --task <TASK-NAME> --model_path /home/deployment
        ```
    """
    return "".join((stub_info, metrics_info, download_instructions))


@click.command(
    context_settings=dict(
        token_normalize_func=lambda x: x.replace("-", "_"),
        show_default=True,
    )
)
@click.version_option(version=__version__)
@click.argument(
    "directory",
    type=str,
    default="",  # defaulting to `None` throws a missing argument Error
)
@click.option(
    "--task",
    type=click.Choice(TASKS_WITH_ALIASES, case_sensitive=False),
    help="The task to find model for, must be specified if `--dataset` not provided",
)
@click.option(
    "--dataset",
    type=click.Choice(DATASETS, case_sensitive=False),
    help="The public dataset used to train this model, must be specified if "
    "`--task` not provided",
)
@click.option(
    "--optimizing-metric",
    "-m",
    default=DEFAULT_OPTIMIZING_METRIC,
    type=str,
    help="The criterion to search model for, multiple metrics can be specified "
    f"as comma separated values, supported metrics are {METRICS}",
    callback=_csv_callback,
)
@click.option(
    "--target",
    type=click.Choice(DEPLOYMENT_SCENARIOS, case_sensitive=False),
    default=DEFAULT_DEPLOYMENT_SCENARIO,
    help="Deployment target scenario (ie 'VNNI' for VNNI capable CPUs)",
    show_default=True,
)
def main(**kwargs):
    r"""
    Utility to fetch a deployment directory for a task based on specified
    optimizing-metric

    Example for using sparsezoo.deployment_package:

         1) `sparsezoo.deployment_package --task image_classification \
                --optimizing_metric accuracy

         2) `sparsezoo.deployment_package --task ic \
                --optimizing_metric "accuracy, compression" \
                --target VNNI`
    """
    if not (kwargs.get("task") or kwargs.get("dataset")):
        raise ValueError("At-least one of the `task` or `dataset` must be specified")
    _LOGGER.debug(f"{kwargs}")
    results = deployment_package(**kwargs)

    print(_get_template(results=results, destination=kwargs.get("directory")))


if __name__ == "__main__":
    main()
