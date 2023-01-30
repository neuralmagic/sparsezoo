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

from sparsezoo.analysis import ImposedSparsificationInfo, NodeInferenceResult


@pytest.mark.parametrize(
    "cls, init_args",
    [
        (NodeInferenceResult, dict(name="Node_A", avg_run_time=10.004, extras={})),
        (
            ImposedSparsificationInfo,
            dict(
                sparsity=0.1,
                sparsity_block_structure="2:4",
                quantization=True,
                recipe=None,
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
