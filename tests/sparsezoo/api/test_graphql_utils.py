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

from typing import Callable, Dict, List

import pytest

from sparsezoo.api.utils import map_keys, to_camel_case, to_snake_case


@pytest.fixture
def test_args():
    return {
        "snake_case_list": [
            "sparse_tag",
            "sub_domain",
            "token_classification",
            "base_cased",
            "model_id",
            "display_name",
            "model_onnx_size_compressed_bytes",
        ],
        "camel_case_list": [
            "sparseTag",
            "subDomain",
            "tokenClassification",
            "baseCased",
            "modelId",
            "displayName",
            "modelOnnxSizeCompressedBytes",
        ],
    }


def test_to_camel_case(test_args: Dict[str, List[str]]):
    snake_case_list = test_args.get("snake_case_list")
    camel_case_list = test_args.get("camel_case_list")

    actual_camel_case_list = [
        to_camel_case(snake_case) for snake_case in snake_case_list
    ]
    assert actual_camel_case_list == camel_case_list


def test_to_snake_case(test_args: Dict[str, List[str]]):
    camel_case_list = test_args.get("camel_case_list")
    snake_case_list = test_args.get("snake_case_list")

    actual_snake_case_list = [
        to_snake_case(camel_case) for camel_case in camel_case_list
    ]
    assert actual_snake_case_list == snake_case_list


@pytest.mark.parametrize(
    "mapper",
    [
        to_camel_case,
        to_snake_case,
    ],
)
def test_map_keys(
    mapper: Callable[[str], str],
    test_args: Dict[str, List[str]],
):
    camel_case_dict = {
        camel_case: camel_case for camel_case in test_args.get("camel_case_list")
    }
    snake_case_dict = {
        snake_case: snake_case for snake_case in test_args.get("snake_case_list")
    }

    if getattr(mapper, "__name__") == "to_snake_case":
        assert list(
            map_keys(dictionary=camel_case_dict, mapper=mapper)
        ) == test_args.get("snake_case_list")

    elif getattr(mapper, "__name__") == "to_camel_case":
        assert list(
            map_keys(dictionary=snake_case_dict, mapper=mapper)
        ) == test_args.get("camel_case_list")
