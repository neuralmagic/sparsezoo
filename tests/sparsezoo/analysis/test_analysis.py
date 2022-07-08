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

# noqa: F811

import pytest

from sparsezoo.analysis import ModelAnalysis
from tests.sparsezoo.analysis.helpers import (
    get_expected_analysis,
    get_generated_analysis,
    get_test_model_names,
)


@pytest.fixture()
def margin_of_error():
    return 0.05


def pytest_generate_tests(metafunc):
    metafunc.parametrize("model_name", get_test_model_names())

def test_analysis(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    def _extract_nested_dict_values(dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                for pair in _extract_nested_dict_values(value):
                    yield (key, *pair)
            else:
                yield (key, value)

    expected_dict = expected_model_analysis.dict()
    expected_dict_values = _extract_nested_dict_values(expected_dict)
    analysis_dict = model_analysis.dict()
    analysis_dict_values = _extract_nested_dict_values(analysis_dict)

    for analysis_dict_value, expected_dict_value in zip(analysis_dict_values, expected_dict_values):
        assert analysis_dict_value == expected_dict_value
