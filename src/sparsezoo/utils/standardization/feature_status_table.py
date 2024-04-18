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

"""
Representation of feature status for a logical grouping of features
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from pydantic.v1 import BaseModel, Field

from sparsezoo.utils.standardization.feature_status import FeatureStatus
from sparsezoo.utils.standardization.markdown_utils import create_markdown_table


__all__ = [
    "FeatureStatusTable",
]


_FEATURE_STATUS_TABLE_MARKDOWN_TEMPLATE = """
## {name}
{description}

{table}
""".strip()


class FeatureStatusTable(ABC, BaseModel):
    """
    Base Pydantic model that includes utilities for building status tables from its
    fields with type FeatureStatus

    Subclasses must define the name property
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: name of feature group this status table represents
        """
        raise NotImplementedError

    @property
    def description(self) -> str:
        """
        :return: description of this feature status group
        """
        return ""

    @property
    def feature_status_fields(self) -> List[Field]:
        """
        :return: Field definitions of fields in this model whose target type is
            FeatureStatus. These fields will be used to build tables
        """
        return [
            field for field in self.__fields__.values() if field.type_ is FeatureStatus
        ]

    def yaml_str_lines(self, indentation: str = "") -> List[str]:
        """
        :param indentation: optional indentation to add to each
        :return: list of lines for a line by line yaml representation of this table
        """
        return [
            f"{indentation}{field.name}: {getattr(self, field.name)}"
            for field in self.feature_status_fields
        ]

    def markdown(self) -> str:
        """
        :return: markdown representation of this table with title. Headers will
            be FeatureStatus property names. Single row will be status for this
            project
        """
        table_headers, table_rows = self.table_header_and_rows()
        table = create_markdown_table(table_headers, table_rows)

        return _FEATURE_STATUS_TABLE_MARKDOWN_TEMPLATE.format(
            name=self.name,
            description=self.description,
            table=table,
        )

    def table_header_and_rows(self) -> Tuple[List[str], List[List[str]]]:
        """
        :return: tuple of markdown table headers and list of rows represented as lists
        """
        feature_status_fields = self.feature_status_fields
        table_headers = [field.name for field in feature_status_fields]
        table_emoji_rows = [
            [
                getattr(self, field.name).github_emoji()
                for field in feature_status_fields
            ]
        ]
        return table_headers, table_emoji_rows

    @staticmethod
    def merged_markdown(
        project_names: List[str],
        status_tables: List["FeatureStatusTable"],
    ) -> str:
        """
        :param project_names: List of names for the projects that the tables represent
        :param status_tables: list of status tables to merge. Must all be instances
            of the same class
        :return: combined markdown table for all given tables with title. Headers
            will be project names. Rows will be feature status
        """
        if not status_tables:
            return ""

        if len(project_names) != len(status_tables):
            raise ValueError(
                f"number of project names does not match number of status tables "
                f"{len(project_names)} != {len(status_tables)}"
            )

        if not all(
            table.__class__ is status_tables[0].__class__ for table in status_tables
        ):
            raise ValueError(
                f"All status tables must be instances of the same class. Found "
                f"classes: {[table.__class__.__name__ for table in status_tables]}"
            )

        table_headers_and_rows = [
            table.table_header_and_rows() for table in status_tables
        ]

        # build rows of feature name + feature statuses per project
        rows = []
        features = table_headers_and_rows[0][0]  # one row per feature
        for row_idx, feature_name in enumerate(features):
            feature_name = f"**{feature_name}**"  # feature name bolded
            statuses = [
                project_row[0][row_idx] for _, project_row in table_headers_and_rows
            ]
            rows.append([feature_name] + statuses)

        # build headers as project names
        headers = [""] + project_names

        table = create_markdown_table(headers, rows)

        return _FEATURE_STATUS_TABLE_MARKDOWN_TEMPLATE.format(
            name=status_tables[0].name,
            description=status_tables[0].description,
            table=table,
        )

    @classmethod
    def default(cls) -> "FeatureStatusTable":
        """
        :return: instance of this class with "n" for every field. for template
            generation only
        """
        default_statuses = {
            field_name: "n"  # default to 'n'
            for field_name, field in cls.__fields__.items()
            if field.type_ is FeatureStatus
        }
        return cls(**default_statuses)
