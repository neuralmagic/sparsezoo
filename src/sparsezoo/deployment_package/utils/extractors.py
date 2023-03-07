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
A file containing extractors for different optimizing metrics
"""

import logging
from types import MappingProxyType

from sparsezoo import Model


_LOGGER = logging.getLogger(__name__)


def _size(model: Model) -> float:
    size = getattr(model, "compressed_size", None)
    if not size:
        raise ValueError(f"Could not determine size for {model}")
    return size


def _throughput(model: Model, num_cores: int = 24, batch_size: int = 64) -> float:
    # num_cores : 24, batch_size: 64 are standard defaults in sparsezoo
    throughput_results = getattr(model, "validation_results", {}).get("throughput", [])

    for throughput_result in throughput_results:
        if (
            throughput_result.batch_size == batch_size
            and throughput_result.num_cores == num_cores
        ):
            return throughput_result.recorded_value

    raise ValueError(
        f"Could not find a throughput result with a specified batch_size of "
        f"{batch_size} and num_cores {num_cores}, for model {model}"
    )


def _batch_size_one_latency(model: Model):
    batch_1_throughput = _throughput(model=model, batch_size=1)

    if not batch_1_throughput:
        raise ValueError(
            f"Could not find a batch size 1 result for calculating latency"
            f" for model {model}"
        )

    return 1000 / batch_1_throughput


def _accuracy(model: Model, metric_name=None) -> float:
    validation_results = getattr(model, "validation_results", {}).get("validation")
    if not validation_results:
        raise ValueError(f"Validation results not found for model {model}")

    if metric_name is not None:
        for result in validation_results:
            if metric_name in result.recorded_units.lower():
                return result.recorded_value
        _LOGGER.info(f"metric name {metric_name} not found for model {model}")

    # fallback to if any accuracy metric found
    accuracy_metrics = ["accuracy", "f1", "recall", "map", "top1 accuracy"]
    for result in validation_results:
        if result.recorded_units.lower() in accuracy_metrics:
            return result.recorded_value

    raise ValueError(
        f"Could not find any accuracy metric {accuracy_metrics} for model {model}"
    )


EXTRACTORS = MappingProxyType(
    {
        "compression": _size,
        "file_size": _size,
        "memory_usage": _size,
        "latency": _batch_size_one_latency,
        "throughput": _throughput,
        "accuracy": _accuracy,
        "f1": _accuracy,
        "map": _accuracy,
        "recall": _accuracy,
    }
)

MINIMIZABLE_METRICS = frozenset({"compression", "file_size", "memory_usage", "latency"})
