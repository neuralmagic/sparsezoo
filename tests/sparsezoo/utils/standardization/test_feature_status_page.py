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

from pydantic.v1 import Field

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

_EXPECTED_TEST_FEATURE_STATUS_TABLE_MERGED_MARKDOWN_OUTPUT = """
# SparseML Fake Project Status Page
fake page description

## fake table 1
fake description

|               | YOLOv5             | Image Classification | Question Answering |
| ------------- | ------------------ | -------------------- | ------------------ |
| **feature_1** | :white_check_mark: | :white_check_mark:   | :white_check_mark: |
| **feature_2** | :question:         | :white_check_mark:   | :heavy_check_mark: |

## fake table 2


|               | YOLOv5             | Image Classification | Question Answering |
| ------------- | ------------------ | -------------------- | ------------------ |
| **feature_3** | :x:                | :white_check_mark:   | :heavy_check_mark: |
| **feature_4** | :heavy_check_mark: | :white_check_mark:   | :white_check_mark: |

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
            feature_2="q",
        ),
        table_2=FakeStatusTable2(
            feature_3="n",
            feature_4="e",
        ),
    )

    assert page.markdown().strip() == (
        _EXPECTED_TEST_FEATURE_STATUS_TABLE_MARKDOWN_OUTPUT.strip()
    )


def test_feature_status_table_merged_markdown():
    page_1 = FakeStatusPage(
        project_name="YOLOv5",
        project_description="Yolo project",
        table_1=FakeStatusTable1(
            feature_1="y",
            feature_2="q",
        ),
        table_2=FakeStatusTable2(
            feature_3="n",
            feature_4="e",
        ),
    )
    page_2 = FakeStatusPage(
        project_name="Image Classification",
        project_description="IC project",
        table_1=FakeStatusTable1(
            feature_1="y",
            feature_2="y",
        ),
        table_2=FakeStatusTable2(
            feature_3="y",
            feature_4="y",
        ),
    )
    page_3 = FakeStatusPage(
        project_name="Question Answering",
        project_description="NLP/QA project",
        table_1=FakeStatusTable1(
            feature_1="y",
            feature_2="e",
        ),
        table_2=FakeStatusTable2(
            feature_3="e",
            feature_4="y",
        ),
    )
    repo_name = "SparseML"

    merged_markdown = FeatureStatusPage.merged_markdown(
        [page_1, page_2, page_3], repo_name
    )

    assert merged_markdown.strip() == (
        _EXPECTED_TEST_FEATURE_STATUS_TABLE_MERGED_MARKDOWN_OUTPUT.strip()
    )


def test_feature_status_table_yaml_serialization():
    page_obj = FakeStatusPage(
        project_name="Fake Project",
        project_description="Fake project status for testing",
        table_1=FakeStatusTable1(
            feature_1="y",
            feature_2="q",
        ),
        table_2=FakeStatusTable2(
            feature_3="n",
            feature_4="e",
        ),
    )

    page_yaml_str = page_obj.yaml_str()
    page_reloaded = FakeStatusPage.from_yaml(page_yaml_str)

    assert page_obj == page_reloaded


_EXPECTED_TEMPLATE_YAML_STR = """
###########################################################
# Status Keys:
# y: yes - implemented by NM
# e: external - implemented by external integration
# n: no - not implemented yet
# q: question - not sure, not tested, or to be investigated
###########################################################

project_name: project name
project_description: description

table_1:
  feature_1: n
  feature_2: n

table_2:
  feature_3: n
  feature_4: n
"""


def test_feature_status_table_template_yaml_str():
    generated_template_yaml_str = FakeStatusPage.template_yaml_str()
    assert generated_template_yaml_str.strip() == _EXPECTED_TEMPLATE_YAML_STR.strip()
