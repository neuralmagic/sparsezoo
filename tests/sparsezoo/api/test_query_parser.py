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
    parser.parse()

    assert parser.operation_body == parsed["operation_body"]
    assert parser.arguments == parsed["arguments"]
    assert parser.fields == parsed["fields"]
