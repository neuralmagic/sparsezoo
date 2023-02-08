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

from pydantic import Field

from sparsezoo.utils.standardization import (
    FeatureStatus,
    FeatureStatusPage,
    FeatureStatusTable,
)


class FakeStatusTable1(FeatureStatusTable):

    feature_1: FeatureStatus = Field()
    feature_2: FeatureStatus = Field()

    @property
    def name(self) -> str:
        return "fake table 1"

    @property
    def description(self) -> str:
        return "fake description"


class FakeStatusTable2(FeatureStatusTable):
    @property
    def name(self) -> str:
        return "fake table 2"

    feature_3: FeatureStatus = Field()
    feature_4: FeatureStatus = Field()


class FakeStatusPage(FeatureStatusPage):
    table_1: FakeStatusTable1 = Field()
    table_2: FakeStatusTable2 = Field()

    @property
    def name(self) -> str:
        return "Fake Project"

    @property
    def description(self) -> str:
        return "fake page description"


_EXPECTED_TEST_FEATURE_STATUS_TABLE_MARKDOWN_OUTPUT = """
# Fake Project Fake Project Status Page
Fake project status for testing

## fake table 1
fake description

| feature_1          | feature_2  |
| ------------------ | ---------- |
| :white_check_mark: | :question: |

## fake table 2


| feature_3 | feature_4          |
| --------- | ------------------ |
| :x:       | :heavy_check_mark: |

### Key
 * :white_check_mark: - implemented by neuralmagic integration
 * :heavy_check_mark: - implemented by underlying integration
 * :x: - not implemented yet
 * :question: - not sure, not tested, or to be investigated
"""


def test_feature_status_table_markdown():
    page = FakeStatusPage(
        project_name="Fake Project",
        project_description="Fake project status for testing",
        table_1=FakeStatusTable1(
            feature_1="y",
            feature_2="?",
        ),
        table_2=FakeStatusTable2(
            feature_3="n",
            feature_4="e",
        ),
    )

    assert page.markdown().strip() == (
        _EXPECTED_TEST_FEATURE_STATUS_TABLE_MARKDOWN_OUTPUT.strip()
    )
