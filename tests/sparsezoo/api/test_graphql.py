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

from typing import Any, Dict, List, Optional

import pytest

from sparsezoo.api import DEFAULT_FIELDS, GraphQLAPI, to_camel_case, to_snake_case


class TestGraphQLAPI(GraphQLAPI):
    def fetch(
        self,
        operation_body: str,
        arguments: Dict[str, str],
        fields: Optional[List[str]] = None,
        url: Optional[str] = None,
    ) -> None:
        operation_body = to_camel_case(operation_body)
        fields = fields or DEFAULT_FIELDS[operation_body]

        reponse_objects = super().fetch(
            operation_body=operation_body,
            arguments=arguments,
            fields=fields,
            url=url,
        )
        for reponse_object in reponse_objects:
            assert all(to_snake_case(field) in reponse_object for field in fields)


@pytest.mark.parametrize(
    "query_args",
    [
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
            "operation_body": "files",
            "arguments": {
                "display_name": "config.json",
            },
            "fields": None,
        },
        {
            "operation_body": "files",
            "arguments": {
                "display_name": "config.json",
            },
            "fields": ["display_name", "download_url", "file_size"],
        },
        {
            "operation_body": "benchmark_results",
            "arguments": {
                "batch_size": 64,
                "device_info": "c6a.12xlarge",
            },
            "fields": ["num_cores", "num_streams", "runtime"],
        },
        {
            "operation_body": "benchmark_results",
            "arguments": {
                "batch_size": 64,
                "device_info": "c6a.12xlarge",
            },
            "fields": None,
        },
        {
            "operation_body": "training_results",
            "arguments": {
                "dataset_type": "downstream",
                "recordedUnits": "F1",
            },
            "fields": [
                "primaryResult",
                "recordedFormat",
                "recordedUnits",
                "recordedValue",
            ],
        },
        {
            "operation_body": "training_results",
            "arguments": {
                "dataset_type": "downstream",
                "recordedUnits": "F1",
            },
            "fields": None,
        },
        {
            "operation_body": "models",
            "arguments": {"stub": "zoo:mobilenet_v2-1.0-imagenet-base"},
            "fields": {"analysis": {"analysisId": None}},
        },
    ],
)
def test_graphql_api_response(query_args: Dict[str, Any]):
    TestGraphQLAPI().fetch(**query_args)
