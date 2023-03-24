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

import pytest

from sparsezoo.api.exceptions import InvalidQueryException, validate_graphql_response


@pytest.fixture
def test_invalid_query_exception():
    return {
        "data": [
            {
                "message": "Cannot query field 'blah' on type 'Query'.",
                "locations": [{"line": 3, "column": 9}],
            }
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


@pytest.mark.xfail(raises=InvalidQueryException)
@pytest.mark.parametrize(
    "response",
    {
        "ddata": [
            {
                "message": "Cannot query field 'blah' on type 'Query'.",
                "locations": [{"line": 3, "column": 9}],
            }
        ],
    },
    {
        "ddata": [
            {
                "message": "Cannot query field 'blah' on type 'Query'.",
                "locations": [{"line": 3, "column": 9}],
            }
        ],
    },
    {
        "ddata": [
            {
                "message": "Cannot query field 'blah' on type 'Query'.",
                "locations": [{"line": 3, "column": 9}],
            }
        ],
    },
)
def test_whatever(response):
    validate_graphql_response(response)
