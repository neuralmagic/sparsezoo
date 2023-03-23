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
from typing import Callable, Optional

from requests.exceptions import HTTPError
from requests.models import Response


logger = logging.getLogger(__name__)


class InvalidQueryValueError(Exception):
    def __init__(self, error_message: Optional[str] = None, *args, **kwargs):
        super().__init__(
            error_message,
            *args,
            **kwargs,
        )


def graphqlapi_exception_handler(fn: Callable) -> Callable:
    @wraps(fn)
    def inner_function(*args, **kwargs):
        try:
            return fn(*args, **kwargs)

        except HTTPError as _err:
            logger.error(f"request exception: {_err}")

        except InvalidQueryValueError as _err:
            logger.error(f"InvalidQueryValueError: {_err}")

    return inner_function


def validate_graphql_response(response: Response, query_body: str) -> None:
    response.raise_for_status()
    response_json = response.json()

    if response_json.get("data") is None and "errors" in response_json:
        raise InvalidQueryValueError(
            error_message=f"{response_json['errors']}\n{query_body}"
        )
