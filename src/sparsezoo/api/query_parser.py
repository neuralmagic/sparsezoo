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

DEFAULT_FILES_FIELDS = ["displayName", "downloadUrl", "fileSize", "fileType", "modelId"]

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

QUERY_BODY = """
    {{
        {operation_body} {arguments}
            {{
               {fields}
            }}
    }}
"""


class QueryParser:
    """Parse the class input arg fields to be used for graphql post requests"""

    def __init__(
        self,
        operation_body: str,
        arguments: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
    ):
        self._operation_body = operation_body
        self._arguments = arguments
        self._fields = fields
        self._query_body = None

        self._parse()

    def _parse(self):
        """Parse to a string compatible with graphql requst body"""

        self._parse_operation_body()
        self._parse_arguments()
        self._parse_fields()
        self._build_query_body()

    def _parse_operation_body(self) -> None:
        self._operation_body = to_camel_case(self._operation_body)

    def _parse_arguments(self) -> None:
        """Transform deprecated stub args and convert to camel case"""
        parsed_arguments = ""
        arguments = self.arguments or {}

        for argument, value in arguments.items():
            if value is not None:
                contemporary_key = DEPRECATED_STUB_ARGS_MAPPER.get(argument, argument)
                camel_case_key = to_camel_case(contemporary_key)

                # single, double quotes matters
                if isinstance(value, str):
                    parsed_arguments += f'{camel_case_key}: "{value}",'
                elif isinstance(value, bool):
                    value = str(value).lower()
                    parsed_arguments += f"{camel_case_key}: {value},"
                else:
                    parsed_arguments += f"{camel_case_key}: {value},"

        if parsed_arguments:
            parsed_arguments = "(" + parsed_arguments + ")"
        self._arguments = parsed_arguments

    def _parse_fields(self) -> None:
        fields = self.fields or DEFAULT_FIELDS.get(self.operation_body)

        field_parsers = {
            List: self.parse_list_fields_to_string,
            Dict: self.parse_dict_fields_to_string,
        }

        for fields_type, parser in field_parsers.items():
            if isinstance(fields, fields_type):
                self.fields = parser(fields)
                break

    def parse_list_fields_to_string(self, fields: List[str]) -> str:
        parsed_fields = ""
        for field in fields:
            field_without_arguments, sep, args = field.partition("(")
            camel_case_field = to_camel_case(field_without_arguments)
            args_str = f"{sep}{args}" if args else ""
            parsed_fields += f"{camel_case_field}{args_str} "
            if camel_case_field in DEFAULT_FIELDS:
                stringified_fields = self.parse_list_fields_to_string(
                    DEFAULT_FIELDS.get(camel_case_field)
                )
                parsed_fields += f"{{ {stringified_fields}}} "

        return parsed_fields

    def parse_dict_fields_to_string(self, fields: Dict[str, Optional[Dict]]) -> str:
        parsed_fields = ""
        for field, field_dict in fields.items():
            if field_dict:
                assert isinstance(field_dict, Dict), (
                    "QueryParser fields must be either dict where ",
                    "values are a dict or None",
                )

                parent_field = to_camel_case(field)
                child_fields = f" {{ {self.parse_dict_fields_to_string(field_dict)}}} "
                parsed_fields += parent_field + child_fields
            else:
                camel_case_field = to_camel_case(field)
                parsed_fields += f"{camel_case_field} "
                if camel_case_field in DEFAULT_FIELDS:
                    fields_for_field = self.parse_list_fields_to_string(
                        DEFAULT_FIELDS.get(camel_case_field)
                    )
                    parsed_fields += f"{{ {fields_for_field}}} "
        return parsed_fields

    def _build_query_body(self) -> None:
        self.query_body = QUERY_BODY.format(
            operation_body=self.operation_body,
            arguments=self.arguments,
            fields=self.fields,
        )

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

    @property
    def query_body(self) -> str:
        """Return the query body"""
        return self._query_body

    @query_body.setter
    def query_body(self, query_body: str) -> None:
        self._query_body = query_body
