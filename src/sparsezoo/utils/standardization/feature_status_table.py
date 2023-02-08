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
from typing import List

from pydantic import BaseModel, Field

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

    def markdown(self) -> str:
        feature_status_fields = self.feature_status_fields
        table_headers = [field.name for field in feature_status_fields]
        table_emoji_rows = [
            [
                getattr(self, field.name).github_emoji()
                for field in feature_status_fields
            ]
        ]
        table = create_markdown_table(table_headers, table_emoji_rows)

        return _FEATURE_STATUS_TABLE_MARKDOWN_TEMPLATE.format(
            name=self.name,
            description=self.description,
            table=table,
        )
