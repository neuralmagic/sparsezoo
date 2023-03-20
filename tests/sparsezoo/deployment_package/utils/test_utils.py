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
Unit tests for `sparsify.package` backend utilities
"""
from typing import List
from unittest.mock import patch

import pytest

from sparsezoo import Model
from sparsezoo.deployment_package.utils.utils import (
    compute_heuristics,
    extract_metrics,
    extract_ranges,
    filter_candidates,
    first_quantized_model,
    get_best_model_with_metrics,
    infer_domain_and_subdomain,
    recommend_stub,
    validate_optimizing_metrics,
)


# fixtures
@pytest.fixture(scope="session")
def quantized_model() -> Model:
    """
    An auto-delete fixture for returning a quantized stub
    """
    yield Model(
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
        "imagenet/pruned95_quant-none"
    )


@pytest.fixture(scope="session")
def vnni_model() -> Model:
    """
    An auto-delete fixture for returning a quantized stub
    """
    yield Model(
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
        "imagenet/pruned85_quant-none-vnni"
    )


@pytest.fixture(scope="session")
def non_quantized_models() -> List[Model]:
    """
    An auto-delete fixture for returning a non-quantized stub
    """

    yield [
        Model(
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
            "imagenette/base-none"
        ),
        Model(
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
            "imagenet/pruned-moderate"
        ),
    ]


@pytest.fixture(scope="session")
def candidates(quantized_model, non_quantized_models, vnni_model) -> List[Model]:
    """
    An auto-delete fixture for returning a list of sparsezoo.Model objects
    from stubs
    """
    models = [quantized_model, vnni_model]
    models.extend(non_quantized_models)
    yield models


# tests


@pytest.mark.parametrize("optimizing_metrics", [["compression", "accuracy", "latency"]])
def test_filter_candidates(
    candidates: List[Model],
    optimizing_metrics: List[str],
):
    filtered_candidates = filter_candidates(candidates, optimizing_metrics)
    assert len(filtered_candidates) <= len(candidates)
    assert all(isinstance(candidate, Model) for candidate in filtered_candidates)


@pytest.mark.parametrize(
    "optimizing_metrics, expected",
    [
        (["A", "B", "C"], ValueError()),
        (
            ["compression", "accuracy", "latency"],
            ["compression", "accuracy", "latency"],
        ),
        (
            ["COMPRESSION", "ACCURACY", "LATENCY"],
            ["compression", "accuracy", "latency"],
        ),
    ],
)
def test_validate_optimizing_metrics(optimizing_metrics, expected):
    if isinstance(expected, ValueError):
        with pytest.raises(ValueError, match="not find a relevant extractor"):
            validate_optimizing_metrics(optimizing_metrics=optimizing_metrics)
    else:
        actual = validate_optimizing_metrics(optimizing_metrics=optimizing_metrics)
        assert actual == expected


def test_first_quantized_model(candidates, quantized_model, non_quantized_models):
    test_cases = [
        (candidates, quantized_model),
        (non_quantized_models, non_quantized_models[0]),
        ([], None),
    ]
    for param, expected in test_cases:
        actual = first_quantized_model(candidates=param)
        assert actual == expected


@pytest.mark.parametrize(
    "optimizing_metrics",
    [
        ["accuracy", "compression"],
        ["accuracy", "compression", "file_size"],
        ["accuracy", "compression", "file_size", "throughput"],
    ],
)
def test_extract_metrics(candidates, optimizing_metrics):
    extracted_metrics = extract_metrics(
        candidates=candidates, optimizing_metrics=optimizing_metrics
    )
    assert len(extracted_metrics) == len(candidates)
    for candidate_metrics in extracted_metrics:
        assert all(
            metric_name in candidate_metrics for metric_name in optimizing_metrics
        )


@pytest.mark.parametrize(
    "extracted_metrics, expected",
    [
        (
            [{"accuracy": 1, "compression": 2}, {"accuracy": 0, "compression": 0}],
            {"accuracy": (0, 1), "compression": (0, 2)},
        ),
        ([{"a": 1, "b": 2}, {"a": 1, "b": 2}], {"a": (1, 1), "b": (2, 2)}),
    ],
)
def test_extract_ranges(extracted_metrics, expected):
    actual_ranges = extract_ranges(extracted_metrics)
    assert actual_ranges == expected


@pytest.mark.parametrize(
    "metrics, ranges, expected",
    [
        (
            {"accuracy": 1, "throughput": 2},
            {"accuracy": (0, 1), "throughput": (0, 2)},
            2,
        ),
        (
            {"accuracy": 1, "throughput": 2},
            {"accuracy": (1, 1), "throughput": (2, 2)},
            0,
        ),
    ],
)
def test_compute_heuristics(metrics, ranges, expected):
    actual = compute_heuristics(metrics, ranges)
    assert actual == expected


@pytest.mark.parametrize(
    "metrics, expected",
    [
        (
            ["accuracy", "compression"],
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/"
            "imagenet/pruned95_quant-none",
        )
    ],
)
def test_get_best_model_with_metrics(candidates, metrics, expected):
    model, _ = get_best_model_with_metrics(
        candidates=candidates,
        optimizing_metrics=metrics,
    )
    actual = model.source
    assert actual == expected


@pytest.mark.parametrize(
    "dataset, task, expected",
    [
        (None, "ic", ("cv", "classification")),
        ("imagenette", "ic", ("cv", "classification")),
        ("blah", "ic", ("cv", "classification")),
        ("imagenet", "blah", ("cv", "classification")),
        ("coco", "object_detection", ("cv", "detection")),
    ],
)
def test_infer_dataset_domain_subdomain(dataset, task, expected):
    assert infer_domain_and_subdomain(dataset=dataset, task=task) == expected


@pytest.mark.parametrize(
    "dataset, task",
    [
        ("mnli", "classification"),
        ("coco", "text_classification"),
        ("blah", "blah"),
    ],
)
def test_infer_dataset_domain_subdomain_raises_value_error(dataset, task):
    with pytest.raises(ValueError):
        infer_domain_and_subdomain(dataset=dataset, task=task)


@patch("sparsezoo.deployment_package_module.utils.utils.get_best_model_with_metrics")
@patch("sparsezoo.deployment_package_module.utils.utils.search_models")
@pytest.mark.parametrize(
    "vnni",
    [
        Model(
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet"
            "/channel20_pruned75_quant-none-vnni"
        )
    ],
)
def test_recommend_model(search_models_func, get_best_model_function, vnni, candidates):
    get_best_model_function.return_value = vnni, {}
    search_models_func.return_value = candidates

    assert recommend_stub(task="ic", optimizing_metrics=["accuracy"])[0] == vnni.source
    assert recommend_stub(task="ic")[0] == vnni.source
    assert recommend_stub(dataset="imagenet")[0] == vnni.source
    assert recommend_stub(dataset="imagenet", scenario="vnni")[0] == vnni.source
    assert recommend_stub(task="qa", dataset="squad", scenario=None)[0] == vnni.source
    assert recommend_stub(task="qa", dataset="squad", scenario="vnni")[0] == vnni.source


@patch("sparsezoo.deployment_package_module.utils.utils.search_models")
def test_value_error_with_recommend_stub(search_models_func, quantized_model):
    # search results empty
    search_models_func.return_value = []
    with pytest.raises(ValueError, match="not find any relevant"):
        recommend_stub(task="ic")

    # deployment scenario not found
    search_models_func.return_value = [quantized_model]
    with pytest.raises(ValueError, match="not find any relevant"):
        recommend_stub(task="ic", scenario="vnni")

    # at-least dataset or task must be specified
    with pytest.raises(ValueError, match="Could not find any info"):
        recommend_stub(scenario="vnni")

    # at-least dataset or task must be specified
    with pytest.raises(ValueError, match="Could not find any info"):
        recommend_stub()
