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

# flake8: noqa

import pytest

from sparsezoo.analyze_v1 import (
    ModelAnalysis,
    draw_operation_chart,
    draw_parameter_chart,
    draw_parameter_operation_combined_chart,
    draw_sparsity_by_layer_chart,
)
from tests.sparsezoo.analyze.helpers import (
    get_expected_analysis,
    get_generated_analysis,
    get_test_model_names,
)


def pytest_generate_tests(metafunc):
    metafunc.parametrize("model_name", get_test_model_names())


def test_draw_sparsity_by_layer_chart(model_name, get_expected_analysis):
    model_analysis = get_expected_analysis(model_name)
    draw_sparsity_by_layer_chart(
        model_analysis,
        model_name=model_name,
    )


def test_draw_operation_chart(model_name, get_expected_analysis):
    model_analysis = get_expected_analysis(model_name)
    draw_operation_chart(
        model_analysis,
        model_name=model_name,
    )


def test_draw_parameter_chart(model_name, get_expected_analysis):
    model_analysis = get_expected_analysis(model_name)
    draw_parameter_chart(
        model_analysis,
        model_name=model_name,
    )


def test_draw_parameter_operation_combined_chart(model_name, get_expected_analysis):
    model_analysis = get_expected_analysis(model_name)
    draw_parameter_operation_combined_chart(
        model_analysis,
        model_name=model_name,
    )
