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

import pytest

from sparsezoo.analyze_v1 import ModelAnalysis
from sparsezoo.analyze_v1.analysis import ModelAnalysisSummary
from sparsezoo.analyze_v1.utils.models import ModelEntry, Section, SizedModelEntry


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
def analysis():
    return ModelAnalysis.create(
        file_path="zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
        "imagenet/pruned_quant-moderate"
    )


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


def test_subtraction(summary_object: ModelAnalysisSummary):
    subtracted_object = summary_object - summary_object
    assert isinstance(subtracted_object, ModelAnalysisSummary)
    for section in subtracted_object.sections:
        for entry in section.entries:
            for field, value in entry.__fields__.items():
                if isinstance(value, Number):
                    assert 0 == value


def test_from_model_analysis(analysis):
    summary = ModelAnalysisSummary.from_model_analysis(analysis=analysis)
    assert isinstance(summary, ModelAnalysisSummary)
    summary.pretty_print()
