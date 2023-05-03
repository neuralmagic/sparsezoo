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

import logging
from functools import wraps
from typing import Callable

from requests.models import Response


logger = logging.getLogger(__name__)


class InvalidQueryException(Exception):
    pass


def graphqlapi_exception_handler(fn: Callable) -> Callable:
    @wraps(fn)
    def inner_function(*args, **kwargs):
        try:
            return fn(*args, **kwargs)

        except InvalidQueryException:
            raise

    return inner_function


def validate_graphql_response(response: Response, query_body: str) -> None:
    response.raise_for_status()
    response_json = response.json()

    if "errors" in response_json:
        raise InvalidQueryException(f"{response_json['errors']}\n{query_body}")
