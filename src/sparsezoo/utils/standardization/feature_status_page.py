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

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field

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

    Subclasses must define the name project type
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

    def markdown(self) -> str:
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
