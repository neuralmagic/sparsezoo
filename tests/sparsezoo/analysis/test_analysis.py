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


def test_layer_counts(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.layer_counts == expected_model_analysis.layer_counts


def test_non_parameterized_operator_counts(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.non_parameterized_operator_counts
        == expected_model_analysis.non_parameterized_operator_counts
    )


def test_num_dense_ops(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_dense_ops == expected_model_analysis.num_dense_ops


def test_num_sparse_ops(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_sparse_ops == expected_model_analysis.num_sparse_ops


def test_num_sparse_layers(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_sparse_layers == expected_model_analysis.num_sparse_layers


def test_num_quantized_layers(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.num_quantized_layers
        == expected_model_analysis.num_quantized_layers
    )


def test_num_parameters(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_parameters == expected_model_analysis.num_parameters


def test_num_sparse_parameters(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.num_sparse_parameters
        == expected_model_analysis.num_sparse_parameters
    )


def test_num_four_blocks(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_four_blocks == expected_model_analysis.num_four_blocks


def test_num_sparse_four_blocks(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert (
        model_analysis.num_sparse_four_blocks
        == expected_model_analysis.num_sparse_four_blocks
    )


def test_average_sparsity(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.average_sparsity == pytest.approx(
        expected_model_analysis.average_sparsity, abs=margin_of_error
    )


def test_average_four_block_sparsity(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.average_four_block_sparsity == pytest.approx(
        expected_model_analysis.average_four_block_sparsity, abs=margin_of_error
    )


def test_num_dense_quantized_ops(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_dense_quantized_ops == pytest.approx(
        expected_model_analysis.num_dense_quantized_ops, abs=margin_of_error
    )


def test_num_sparse_quantized_ops(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_sparse_quantized_ops == pytest.approx(
        expected_model_analysis.num_sparse_quantized_ops, abs=margin_of_error
    )


def test_num_sparse_floating_point_ops(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_sparse_floating_point_ops == pytest.approx(
        expected_model_analysis.num_sparse_floating_point_ops, abs=margin_of_error
    )


def test_num_dense_floating_point_ops(
    model_name, get_generated_analysis, get_expected_analysis
):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert model_analysis.num_dense_floating_point_ops == pytest.approx(
        expected_model_analysis.num_dense_floating_point_ops, abs=margin_of_error
    )


def test_node_analyses(model_name, get_generated_analysis, get_expected_analysis):
    model_analysis = get_generated_analysis(model_name)
    expected_model_analysis = get_expected_analysis(model_name)

    assert len(model_analysis.nodes) == len(expected_model_analysis.nodes)
    for i in range(len(model_analysis.nodes)):
        node_analysis = model_analysis.nodes[i]
        expected_node_analysis = expected_model_analysis.nodes[i]
        assert node_analysis == expected_node_analysis


def test_model_analysis_yaml(model_name, get_generated_analysis):
    model_analysis = get_generated_analysis(model_name)

    model_yaml = model_analysis.yaml()
    model_from_yaml = ModelAnalysis.parse_yaml_raw(model_yaml)

    assert model_analysis == model_from_yaml
