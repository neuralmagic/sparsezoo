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

from typing import Dict
from unittest.mock import MagicMock

import pytest

from sparsezoo.api.exceptions import InvalidQueryException, validate_graphql_response


@pytest.mark.parametrize(
    "should_raise_invalid_query_exception,response",
    [
        (
            True,
            {
                "data": None,
                "errors": {
                    "message": "Cannot query field 'blah' on type 'Query'.",
                    "locations": [{"line": 3, "column": 9}],
                },
            },
        ),
        (
            True,
            {
                "data": None,
                "errors": {
                    "message": "Cannot query field 'foo' on type 'Query'.",
                    "locations": [{"line": 5, "column": 9}],
                },
            },
        ),
        (
            False,
            {
                "data": "foo",
            },
        ),
    ],
)
def test_graphql_error_response(
    should_raise_invalid_query_exception: bool, response: Dict
):
    mock_response = MagicMock()
    mock_response.json.return_value = response

    if should_raise_invalid_query_exception:
        with pytest.raises(InvalidQueryException):
            validate_graphql_response(mock_response, "")
    else:
        validate_graphql_response(mock_response, "")
