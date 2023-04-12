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

import pytest

from sparsezoo.utils import TASK_REGISTRY, TaskName


@pytest.mark.parametrize(
    "name, aliases, domain, sub_domain, test_aliases, test_not_aliases",
    [
        (
            "image_classification",
            ["ic", "classification"],
            "cv",
            "classification",
            ["Image-Classification", "ic", "IC", "classification", "CLASSIFICATION"],
            ["image classification", "imageclassification"],
        ),
        (
            "object_detection",
            ["od", "detection"],
            "cv",
            "detection",
            [
                "Object-Detection",
                "od",
                "OD",
                "detection",
                "DETECTION",
            ],
            ["object detection", "objectdetection"],
        ),
        (
            "segmentation",
            [],
            "cv",
            "segmentation",
            [
                "Segmentation",
                "SEGMENTATION",
            ],
            ["object_segmentation"],
        ),
        (
            "question_answering",
            ["qa"],
            "nlp",
            "question_answering",
            [
                "Question-Answering",
                "qa",
                "QA",
            ],
            ["question answering", "questionanswering"],
        ),
        (
            "text_classification",
            ["glue"],
            "nlp",
            "text_classification",
            [
                "Text-Classification",
                "glue",
                "GLUE",
            ],
            ["text classification", "textclassification"],
        ),
        (
            "sentiment_analysis",
            ["sentiment"],
            "nlp",
            "sentiment_analysis",
            [
                "Sentiment-Analysis",
                "sentiment",
                "SENTIMENT",
            ],
            ["sentiment analysis", "sentimentanalysis"],
        ),
        (
            "token_classification",
            ["ner", "named_entity_recognition"],
            "nlp",
            "token_classification",
            ["Token-Classification", "ner", "NER", "Named-Entity-Recognition"],
            ["token classification", "tokenclassification"],
        ),
    ],
)
class TestTaskName:
    def test_task_create(
        self,
        name,
        aliases,
        domain,
        sub_domain,
        test_aliases,
        test_not_aliases,
    ):
        task = TaskName(name, domain, sub_domain, aliases)

        assert task == name
        assert task.domain == domain
        assert task.sub_domain == sub_domain
        assert TASK_REGISTRY[name] == task

        for alias in aliases + test_aliases:
            assert alias == task

        for alias in test_not_aliases:
            assert alias != task

    def test_task_add_alias(
        self,
        name,
        aliases,
        domain,
        sub_domain,
        test_aliases,
        test_not_aliases,
    ):
        task = TaskName(name, domain, sub_domain, aliases)

        for alias in test_not_aliases:
            task._add_alias(alias)
            assert alias == task
            assert alias.upper() == task
            assert alias.lower() == task
            assert alias.replace("_", "-") == task
            assert alias.replace("-", "_") == task

        for alias in aliases + test_aliases:
            assert alias == task

    def test_task_immutability(
        self,
        name,
        aliases,
        domain,
        sub_domain,
        test_aliases,
        test_not_aliases,
    ):
        task = TaskName(name, domain, sub_domain, aliases)

        for field in ["name", "domain", "sub_domain", "aliases"]:
            try:
                setattr(task, field, "")
                pytest.fail(
                    f"TaskName is not immutable. '{field}' was successfully modified"
                )
            except AttributeError:
                continue
