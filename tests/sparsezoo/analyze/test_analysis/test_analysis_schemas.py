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

import pytest
import yaml

from sparsezoo.analyze_v1 import (
    BenchmarkResult,
    BenchmarkScenario,
    ImposedSparsificationInfo,
    NodeInferenceResult,
)


def _imposed_sparsification_args():
    return dict(
        sparsity=0.1,
        sparsity_block_structure="2:4",
        quantization=True,
        recipe=None,
    )


def _benchmark_setup_args():
    return dict(
        batch_size=4,
        num_cores=None,
        engine="deepsparse",
        scenario="sync",
        num_streams=None,
        duration=10,
        warmup_duration=10,
        instructions=None,
        analysis_only=False,
    )


def _node_inference_result_args():
    return dict(name="Node_A", avg_run_time=10.004, extras={})


@pytest.mark.parametrize(
    "cls, init_args",
    [
        (NodeInferenceResult, _node_inference_result_args()),
        (ImposedSparsificationInfo, _imposed_sparsification_args()),
        (BenchmarkScenario, _benchmark_setup_args()),
        (
            BenchmarkResult,
            dict(
                setup=BenchmarkScenario(**_benchmark_setup_args()),
                imposed_sparsification=ImposedSparsificationInfo(
                    **_imposed_sparsification_args(),
                ),
                items_per_second=1.5,
                average_latency=666.67,
                node_timings=[NodeInferenceResult(**_node_inference_result_args())],
            ),
        ),
    ],
)
def test_yaml_serialization(
    cls,
    init_args,
):
    expected_results = cls(**init_args)
    args_from_yaml = yaml.safe_load(expected_results.yaml())
    actual_results = cls(**args_from_yaml)

    for arg in init_args:
        assert getattr(expected_results, arg) == getattr(actual_results, arg)
