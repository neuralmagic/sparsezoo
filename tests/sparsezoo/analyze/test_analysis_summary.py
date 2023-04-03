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
from numbers import Number
from pathlib import Path

import pytest

from sparsezoo.analyze.analysis_summary import (
    ModelAnalysisSummary,
    ModelEntry,
    Section,
    SizedModelEntry,
)


@pytest.fixture
def summary_object():
    yield ModelAnalysisSummary(
        sections=[
            Section(
                section_name="Params",
                entries=[
                    SizedModelEntry(
                        model="model_1.onnx",
                        count=1234,
                        size=4321,
                        sparsity=90.453,
                        quantized=20.1,
                    ),
                    SizedModelEntry(
                        model="model_2.onnx",
                        count=1234,
                        size=4321,
                        sparsity=90.453,
                        quantized=20.1,
                    ),
                    SizedModelEntry(
                        model="model_3.onnx",
                        count=1234,
                        size=4321,
                        sparsity=90.453,
                        quantized=20.1,
                    ),
                ],
            ),
            Section(
                section_name="Ops",
                entries=[
                    SizedModelEntry(
                        model="model_1.onnx",
                        count=1234,
                        size=4321,
                        sparsity=90.453,
                        quantized=20.1,
                    ),
                    SizedModelEntry(
                        model="model_2.onnx",
                        count=1234,
                        size=4321,
                        sparsity=90.453,
                        quantized=20.1,
                    ),
                    SizedModelEntry(
                        model="model_3.onnx",
                        count=1234,
                        size=4321,
                        sparsity=90.453,
                        quantized=20.1,
                    ),
                ],
            ),
            Section(
                section_name="Overall",
                entries=[
                    ModelEntry(model="model_1.onnx", sparsity=90.453, quantized=20.1)
                ],
            ),
        ]
    )


@pytest.fixture
def summary_object_with_expected_pretty_print(summary_object: ModelAnalysisSummary):
    file_path = Path(__file__).parent / "analysis_summary.txt"
    with open(file_path) as f:
        expected_pretty_print = "\n".join(line.strip() for line in f if line.strip())

    return summary_object, expected_pretty_print


def test_yaml_serialization(summary_object: ModelAnalysisSummary):
    expected_yaml = summary_object.yaml()
    yaml_generated_object = ModelAnalysisSummary.parse_yaml_raw(expected_yaml)
    actual_yaml = yaml_generated_object.yaml()
    assert expected_yaml == actual_yaml


def test_if_pretty_printable(summary_object: ModelAnalysisSummary):
    assert hasattr(summary_object, "pretty_print")
    pretty_print_func = getattr(summary_object, "pretty_print")

    assert callable(pretty_print_func)
    summary_object.pretty_print()


def test_pretty_print_output(summary_object_with_expected_pretty_print, capfd):
    summary_object, expected_out = summary_object_with_expected_pretty_print
    summary_object.pretty_print()

    actual_output, err = capfd.readouterr()
    actual_output = "\n".join(
        line.strip() for line in actual_output.split("\n") if line.strip()
    )

    assert actual_output == expected_out


def test_subtraction(summary_object: ModelAnalysisSummary):
    subtracted_object = summary_object - summary_object
    assert isinstance(subtracted_object, ModelAnalysisSummary)
    for section in subtracted_object.sections:
        for entry in section.entries:
            for field, value in entry.__fields__.items():
                if isinstance(value, Number):
                    assert 0 == value
