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


from sparsezoo.utils.constants import TaskName, get_dataset_info, get_task_info


_IC_TASK = TaskName(
    name="image_classification",
    aliases=["ic", "classification"],
    domain="cv",
    sub_domain="classification",
)

_TEXT_CLASSIFICATION_TASK = TaskName(
    name="text_classification",
    aliases=["glue"],
    domain="nlp",
    sub_domain="text_classification",
)


def test_get_task_info():
    assert get_task_info("ic") == _IC_TASK
    assert get_task_info("image_Classification") == _IC_TASK
    assert get_task_info("classification") == _IC_TASK
    assert get_task_info(None) is None


def test_get_dataset_info():
    assert get_dataset_info("mnli") == _TEXT_CLASSIFICATION_TASK
    assert get_dataset_info("MnLi") == _TEXT_CLASSIFICATION_TASK
    assert get_dataset_info("  mNli ") == _TEXT_CLASSIFICATION_TASK
    assert get_dataset_info(None) is None
