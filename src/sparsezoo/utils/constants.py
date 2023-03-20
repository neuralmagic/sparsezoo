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
from typing import Dict, Optional

from sparsezoo.utils import TaskName


__all__ = [
    "DATASETS",
    "DATASET_REGISTRY",
    "DEFAULT_DEPLOYMENT_SCENARIO",
    "DEFAULT_OPTIMIZING_METRIC",
    "DEPLOYMENT_SCENARIOS",
    "METRICS",
    "TASKS",
    "TASK_REGISTRY",
    "TASKS_WITH_ALIASES",
    "get_dataset_info",
    "get_task_info",
]

DEFAULT_OPTIMIZING_METRIC = "accuracy"
METRICS = [
    DEFAULT_OPTIMIZING_METRIC,
    "f1",
    "recall",
    "mAP",
    "latency",
    "throughput",
    "compression",
    "file_size",
    "memory_usage",
]

DEFAULT_DEPLOYMENT_SCENARIO = "DEFAULT"
DEPLOYMENT_SCENARIOS = [
    "VNNI",
    DEFAULT_DEPLOYMENT_SCENARIO,
]

TASK_REGISTRY: Dict[str, TaskName] = {
    "image_classification": TaskName(
        name="image_classification",
        aliases=["ic", "classification"],
        domain="cv",
        sub_domain="classification",
    ),
    "object_detection": TaskName(
        name="object_detection",
        aliases=["od", "detection"],
        domain="cv",
        sub_domain="detection",
    ),
    "segmentation": TaskName(
        name="segmentation", domain="cv", sub_domain="segmentation"
    ),
    "question_answering": TaskName(
        name="question_answering",
        aliases=["qa"],
        domain="nlp",
        sub_domain="question_answering",
    ),
    "text_classification": TaskName(
        name="text_classification",
        aliases=["glue"],
        domain="nlp",
        sub_domain="text_classification",
    ),
    "sentiment_analysis": TaskName(
        name="sentiment_analysis",
        aliases=["sentiment"],
        domain="nlp",
        sub_domain="sentiment_analysis",
    ),
    "token_classification": TaskName(
        name="token_classification",
        aliases=["ner", "named_entity_recognition"],
        domain="nlp",
        sub_domain="token_classification",
    ),
}

DATASET_REGISTRY: Dict[str, TaskName] = {
    "imagenette": TASK_REGISTRY["image_classification"],
    "imagenet": TASK_REGISTRY["image_classification"],
    "coco": TASK_REGISTRY["object_detection"],
    "squad": TASK_REGISTRY["question_answering"],
    "mnli": TASK_REGISTRY["text_classification"],
    "qqp": TASK_REGISTRY["text_classification"],
    "sst2": TASK_REGISTRY["text_classification"],
    "conll2003": TASK_REGISTRY["token_classification"],
}


def get_task_info(task_name: Optional[str]) -> Optional[TaskName]:
    """
    :param task_name: The task name to get information for
    :return: A TaskName object if information found else None
    """
    task_info: Optional[TaskName] = TASK_REGISTRY.get(task_name)

    if task_info:
        return task_info

    # search in aliases
    for name, current_task_info in TASK_REGISTRY.items():
        if task_name == current_task_info:
            return current_task_info


def get_dataset_info(dataset_name: Optional[str]) -> Optional[TaskName]:
    """
    :param dataset_name: The dataset name to get information for
    :return: A TaskName object if information found else None
    """
    if dataset_name:
        dataset_name = dataset_name.lower().strip().replace("-", "_")
    return DATASET_REGISTRY.get(dataset_name)


TASKS = list(TASK_REGISTRY.keys())
DATASETS = list(DATASET_REGISTRY.keys())

TASKS_WITH_ALIASES = []

for task in TASKS:
    TASKS_WITH_ALIASES.append(task)
    TASKS_WITH_ALIASES.extend(TASK_REGISTRY[task].aliases)
