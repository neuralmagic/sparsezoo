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
Page containing a collection of feature status tables. Useful for reporting
project(s) feature status with groups and descriptions
"""

import os
from abc import ABC, abstractmethod
from typing import List

import yaml
from pydantic.v1 import BaseModel, Field

from sparsezoo.utils.standardization.feature_status import FeatureStatus
from sparsezoo.utils.standardization.feature_status_table import FeatureStatusTable


__all__ = [
    "FeatureStatusPage",
]


_FEATURE_STATUS_PAGE_MARKDOWN_TEMPLATE = """
# {project_name} {name} Status Page
{project_description}

{tables}

{status_key}
""".strip()


class FeatureStatusPage(ABC, BaseModel):
    """
    Base Pydantic model that represents a project status page as a series of feature
    status tables

    Subclasses must define the project name
    """

    project_name: str = Field(description="name of this project")
    project_description: str = Field(
        description="Description of this project",
        default="",
    )

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: name of project type this feature status page represents
        """
        raise NotImplementedError

    @property
    def description(self) -> str:
        """
        :return: description of this feature status page
        """
        return ""

    @property
    def feature_status_table_fields(self) -> List[Field]:
        """
        :return: Field definitions of fields in this model whose target type is
            FeatureStatusTable. These fields will become tables on this page
        """
        return [
            field
            for field in self.__fields__.values()
            if issubclass(field.type_, FeatureStatusTable)
        ]

    @classmethod
    def from_yaml(cls, yaml_str_or_path: str) -> "FeatureStatusPage":
        """
        :param yaml_str_or_path: yaml string or path to yaml file to load
        :return: status page object loaded from yaml
        """
        if os.path.exists(yaml_str_or_path):
            with open(yaml_str_or_path, "r") as yaml_file:
                yaml_obj = yaml.safe_load(yaml_file)
        else:
            yaml_obj = yaml.safe_load(yaml_str_or_path)
        return cls.parse_obj(yaml_obj)

    def yaml_str(self) -> str:
        """
        :return: yaml reprsentation of this status page with one line per feature
        """
        yaml_str_lines = list()

        # status key help txt
        yaml_str_lines.append(FeatureStatus.YAML_HELP)
        yaml_str_lines.append("\n")

        # base fields
        yaml_str_lines.append(f"project_name: {self.project_name}")
        yaml_str_lines.append(f"project_description: {self.project_description}")

        # append feature status tables with status indentations of '  '
        for field in self.feature_status_table_fields:
            yaml_str_lines.append("\n")
            yaml_str_lines.append(f"{field.name}:")
            status_table = getattr(self, field.name)
            yaml_str_lines.extend(status_table.yaml_str_lines(indentation="  "))

        return "\n".join(yaml_str_lines).replace("\n\n\n", "\n\n")

    def markdown(self) -> str:
        """
        :return: page represented in markdown format
        """
        feature_status_table_fields = self.feature_status_table_fields
        feature_status_table_markdowns = [
            getattr(self, field.name).markdown()
            for field in feature_status_table_fields
        ]

        return _FEATURE_STATUS_PAGE_MARKDOWN_TEMPLATE.format(
            project_name=self.project_name,
            name=self.name,
            project_description=self.project_description,
            tables="\n\n".join(feature_status_table_markdowns),
            status_key=FeatureStatus.MARKDOWN_HELP,
        )

    @staticmethod
    def merged_markdown(
        status_pages: List["FeatureStatusPage"],
        repo_name: str = "",
    ) -> str:
        """
        :param status_pages: list of status pages to merge. Must all be instances
            of the same class
        :param repo_name: optional repo name to add to front of page title
        :return: combined markdown page for all given pages
        """
        if not status_pages:
            return ""

        if not all(
            page.__class__ is status_pages[0].__class__ for page in status_pages
        ):
            raise ValueError(
                f"All status pages must be instances of the same class. Found "
                f"classes: {[page.__class__.__name__ for page in status_pages]}"
            )

        project_names = [page.project_name for page in status_pages]
        table_field_names = [
            field.name for field in status_pages[0].feature_status_table_fields
        ]
        table_markdowns = []
        for table_field_name in table_field_names:
            status_tables = [
                getattr(status_page, table_field_name) for status_page in status_pages
            ]
            table_markdowns.append(
                FeatureStatusTable.merged_markdown(project_names, status_tables)
            )

        return _FEATURE_STATUS_PAGE_MARKDOWN_TEMPLATE.format(
            project_name=repo_name,
            name=status_pages[0].name,
            project_description=status_pages[0].description,
            tables="\n\n".join(table_markdowns),
            status_key=FeatureStatus.MARKDOWN_HELP,
        )

    @classmethod
    def default(cls) -> "FeatureStatusPage":
        """
        :return: default sattus page with all status values set to 'n'
        """
        default_constructor_args = {
            field_name: field.type_.default()
            for field_name, field in cls.__fields__.items()
            if issubclass(field.type_, FeatureStatusTable)
        }
        default_constructor_args["project_name"] = "project name"
        default_constructor_args["project_description"] = "description"
        return cls(**default_constructor_args)

    @classmethod
    def template_yaml_str(cls) -> str:
        """
        :return: sample yaml string for this class with all status values set to 'n'
        """
        return cls.default().yaml_str()
