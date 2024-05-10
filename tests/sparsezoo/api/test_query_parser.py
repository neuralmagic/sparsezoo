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

from typing import Any, Dict

import pytest

from sparsezoo import QueryParser


@pytest.mark.parametrize(
    "raw,parsed",
    [
        (
            {
                "operation_body": "models",
                "arguments": {
                    "domain": "nlp",
                    "sub_domain": "question_answering",
                },
                "fields": None,
            },
            {
                "operation_body": "models",
                "arguments": '(domain: "nlp",task: "question_answering",)',
                "fields": "modelId stub ",
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": {
                    "domain": "nlp",
                    "sub_domain": "token_classification",
                    "architecture": "biobert",
                    "sub_architecture": "base_cased",
                    "framework": "pytorch",
                    "repo": "huggingface",
                    "dataset": "bc2gm",
                    "sparse_tag": "pruned90-none",
                },
                "fields": [
                    "model_id",
                    "model_onnx_size_compressed_bytes",
                    "stub",
                    "sparse_tag",
                ],
            },
            {
                "operation_body": "models",
                "arguments": (
                    '(domain: "nlp",task: "token_classification",architecture: '
                    '"biobert",subArchitecture: "base_cased",framework: '
                    '"pytorch",repo: "huggingface",sourceDataset: '
                    '"bc2gm",sparseTag: "pruned90-none",)'
                ),
                "fields": "modelId modelOnnxSizeCompressedBytes stub sparseTag ",
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": None,
                "fields": None,
            },
            {
                "operation_body": "models",
                "arguments": "",
                "fields": "modelId stub ",
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": None,
                "fields": {"model_id": None, "display_name": None},
            },
            {
                "operation_body": "models",
                "arguments": "",
                "fields": "modelId displayName ",
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": None,
                "fields": {
                    "model_id": None,
                    "display_name": None,
                    "benchmark_results": {
                        "benchmark_result_id": None,
                        "scenario": None,
                    },
                    "files": {"checkpoint": None},
                },
            },
            {
                "operation_body": "models",
                "arguments": "",
                "fields": (
                    "modelId displayName benchmarkResults "
                    "{ benchmarkResultId scenario } files { checkpoint } "
                ),
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": None,
                "fields": {
                    "model_id": None,
                    "display_name": None,
                    "benchmark_results": None,
                    "files": {"checkpoint": None},
                },
            },
            {
                "operation_body": "models",
                "arguments": "",
                "fields": (
                    "modelId displayName benchmarkResults "
                    "{ batchSize deviceInfo numCores recordedUnits recordedValue } "
                    "files { checkpoint } "
                ),
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": None,
                "fields": ["model_id", "display_name", "benchmark_results", "files"],
            },
            {
                "operation_body": "models",
                "arguments": "",
                "fields": (
                    "modelId displayName benchmarkResults "
                    "{ batchSize deviceInfo numCores recordedUnits recordedValue } "
                    "files { displayName downloadUrl fileSize fileType modelId } "
                ),
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": {"is_publicly_visible": False},
                "fields": {
                    "model_id": None,
                    "similar_models_repo_names": {
                        "display_name": None,
                        "repo_namespace": None,
                    },
                },
            },
            {
                "operation_body": "models",
                "arguments": "(isPubliclyVisible: false,)",
                "fields": (
                    "modelId similarModelsRepoNames { displayName repoNamespace } "
                ),
            },
        ),
        (
            {
                "operation_body": "models",
                "arguments": {"stub": "zoo:mobilenet_v2-1.0-imagenet-base"},
                "fields": {"analysis": {"analysisId": None}},
            },
            {
                "operation_body": "models",
                "arguments": '(stub: "zoo:mobilenet_v2-1.0-imagenet-base",)',
                "fields": "analysis { analysisId } ",
            },
        ),
    ],
)
def test_query_parser(
    raw: Dict[str, Any],
    parsed: Dict[str, Any],
) -> None:
    parser = QueryParser(
        operation_body=raw["operation_body"],
        arguments=raw["arguments"],
        fields=raw["fields"],
    )

    assert parser.operation_body == parsed["operation_body"]
    assert parser.arguments == parsed["arguments"]
    assert parser.fields == parsed["fields"]
