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
from typing import Any, Callable, Dict, List

from requests.models import Response


logger = logging.getLogger(__name__)

DEFAULT_FILE_DISPLAY_NAMES = {
    "model.md",
    "model.onnx",
    "model.onnx.tar.gz",
    "training",
    "deployment",
    "benchmark.yaml",
    "metrics.yaml",
    "analysis.yaml",
}


class InvalidQueryException(Exception):
    pass


class FilesNotFoundException(Exception):
    pass


class DefaultFilesMissingException(Exception):
    pass


def graphqlapi_exception_handler(fn: Callable) -> Callable:
    @wraps(fn)
    def inner_function(*args, **kwargs):
        try:
            response = fn(*args, **kwargs)
            _validate_response_files(response, **kwargs)
            return response

        except InvalidQueryException:
            raise
        except FilesNotFoundException:
            raise
        except DefaultFilesMissingException:
            raise

    return inner_function


def validate_graphql_response(response: Response, query_body: str) -> None:
    response.raise_for_status()
    response_json = response.json()

    if "errors" in response_json:
        raise InvalidQueryException(f"{response_json['errors']}\n{query_body}")


def _validate_response_files(
    response: List[Dict[str, Any]], **kwargs: Dict[str, Any]
) -> None:
    fields: List[str] = kwargs.get("fields")
    if fields is not None and "files" in fields:
        for response_dict in response:
            files: List[Dict[str, Any]] = response_dict.get("files")
            if len(files) == 0:
                raise FilesNotFoundException(f"No files found for {kwargs}")

            file_names = set()
            for file in files:
                file_names.add(file.get("displayName"))

            diff = DEFAULT_FILE_DISPLAY_NAMES.difference(file_names)
            if len(diff) > 0:
                raise DefaultFilesMissingException(
                    f"The following files are missing: {diff} for {kwargs}"
                )
