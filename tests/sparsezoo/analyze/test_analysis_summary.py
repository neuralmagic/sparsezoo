import pytest

from sparsezoo.analyze.analysis_summary import ModelAnalysisSummary, NamedEntry, \
    TypedEntry, SizedModelEntry, Section


@pytest.fixture
def summary_object():
    yield ModelAnalysisSummary(
        sections=[
            Section(
                entries=[
                    NamedEntry(name="Conv2d", sparsity=90.0, quantized=25.0,
                               total=350,
                               size=12345),
                    TypedEntry(type="Weight", sparsity=90.0, quantized=25.0,
                               size=12345),
                    SizedModelEntry(
                        model="dummy.onnx", size=9999,
                                    sparsity=95.5,
                                    quantized=25.0, count=2500)
                ]
            )
        ]
    )


def test_yaml_serialization(summary_object: ModelAnalysisSummary):
    expected_yaml = summary_object.yaml()
    yaml_generated_object = ModelAnalysisSummary.parse_yaml_raw(
        expected_yaml)
    actual_yaml = yaml_generated_object.yaml()
    assert (
            expected_yaml == actual_yaml
    )
