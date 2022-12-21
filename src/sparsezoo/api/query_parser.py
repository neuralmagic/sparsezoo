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


from typing import Dict, List, Optional

from .utils import to_camel_case


DEFAULT_MODELS_FIELDS = ["modelId", "stub"]

DEFAULT_FILES_FIELDS = [
    "displayName",
    "fileSize",
    "modelId",
]

DEFAULT_TRAINING_RESULTS_FIELDS = [
    "datasetName",
    "datasetType",
    "recordedUnits",
    "recordedValue",
]

DEFAULT_BENCHMARK_RESULTS_FIELDS = [
    "batchSize",
    "deviceInfo",
    "numCores",
    "recordedUnits",
    "recordedValue",
]

DEPRECATED_STUB_ARGS_MAPPER = {"sub_domain": "task", "dataset": "source_dataset"}
DEFAULT_FIELDS = {
    "models": DEFAULT_MODELS_FIELDS,
    "files": DEFAULT_FILES_FIELDS,
    "trainingResults": DEFAULT_TRAINING_RESULTS_FIELDS,
    "benchmarkResults": DEFAULT_BENCHMARK_RESULTS_FIELDS,
}


class QueryParser:
    """Parse the class input arg fields to be used for graphql post requests"""

    def __init__(
        self,
        operation_body: str,
        arguments: Dict[str, str],
        fields: Optional[List[str]] = None,
    ):
        self._operation_body = operation_body
        self._arguments = arguments
        self._fields = fields

    def parse(self):
        """Parse to a string compatible with graphql requst body"""

        self._parse_operation_body()
        self._parse_arguments()
        self._parse_fields()

    def _parse_operation_body(self) -> None:
        self._operation_body = to_camel_case(self._operation_body)

    def _parse_arguments(self) -> None:
        """Transform deprecated stub args and convert to camel case"""
        parsed_arguments = ""
        for key, value in self.arguments.items():
            if value is not None:
                contemporary_key = DEPRECATED_STUB_ARGS_MAPPER.get(key, key)
                camel_case_key = to_camel_case(contemporary_key)

                # single, double quotes matters
                parsed_arguments += f'{camel_case_key}: "{value}",'

        if bool(parsed_arguments):
            parsed_arguments = "(" + parsed_arguments + ")"

        self._arguments = parsed_arguments

    def _parse_fields(self) -> None:
        fields = self.fields or DEFAULT_FIELDS.get(self.operation_body)

        parsed_fields = ""
        for field in fields:
            camel_case_field = to_camel_case(field)
            parsed_fields += rf"{camel_case_field} "
        self.fields = parsed_fields

    @property
    def operation_body(self) -> str:
        """Return the query operation body"""
        return self._operation_body

    @operation_body.setter
    def operation_body(self, operation_body: str) -> None:
        self._operation_body = operation_body

    @property
    def arguments(self) -> str:
        """Return the query arguments"""
        return self._arguments

    @arguments.setter
    def arguments(self, arguments: str) -> None:
        self._operation_body = arguments

    @property
    def fields(self) -> str:
        """Return the query fields"""
        return self._fields

    @fields.setter
    def fields(self, fields: str) -> None:
        self._fields = fields


