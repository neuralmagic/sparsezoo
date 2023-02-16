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
Utilities for sparsezoo.deployment_package backend
"""
import logging
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

from sparsezoo import Model, search_models
from sparsezoo.deployment_package.utils.extractors import (
    EXTRACTORS,
    MINIMIZABLE_METRICS,
)
from sparsezoo.utils import TaskName, get_dataset_info, get_task_info


_LOGGER = logging.getLogger(__name__)


def validate_optimizing_metrics(optimizing_metrics: List[str]) -> List[str]:
    """
    A function to normalize all optimizing metric names to lowercase strings
    and validate if they are appropriate names

    :param optimizing_metrics: list of optimizing metric names to normalize and
        validate
    :return: list of valid normalized optimizing metric names
    :post-condition: each optimizing metric name in the returned list will have an
        extractor
    """
    normalized_metric_names: List[str] = [
        metric.lower() for metric in optimizing_metrics
    ]

    # validate metric names
    for metric_name in normalized_metric_names:
        if metric_name not in EXTRACTORS:
            raise ValueError(
                "Could not find a relevant extractor for specified metric name"
                f"{metric_name}"
            )
    return normalized_metric_names


def filter_candidates(
    candidates: List[Model], optimizing_metrics: Optional[List[str]] = None
) -> List[Model]:
    """
    A function to remove candidates which do not have information on needed
    optimizing metrics, additionally logs which Models do not have this info

    :pre-condition: All metric names have been validated to have relevant
        extractors
    :param candidates: list of Model candidates to filter
    :param optimizing_metrics: a list of optimizing metrics to look for in each
        candidate
    :return: A filtered list of models that contain all relevant optimizing
        metrics
    """
    filtered_candidates: List[Model] = []
    for candidate in candidates:
        found_all_metrics = True
        for metric in optimizing_metrics:
            extractor = EXTRACTORS[metric]
            try:
                extractor(candidate)
            except (AttributeError, ValueError, ZeroDivisionError) as exception:
                _LOGGER.info(
                    f"{metric} information could not be extracted from "
                    f"candidate stub {candidate.source}, {exception}"
                )
                found_all_metrics = False
        if found_all_metrics:
            filtered_candidates.append(candidate)

    return filtered_candidates


def first_quantized_model(candidates: List[Model]) -> Optional[Model]:
    """
    A function that finds and returns the first quantized model from a list of
    candidates, if no such model is found then it returns the first model in the
    list

    :param candidates: a list of sparsezoo.Model objects representing
        candidates for finding the first quantized model
    :return: the first quantized model among the candidates if any, else the
        first candidate
    """

    for model in candidates:
        if "quant" in model.source:
            return model
    return candidates[0] if candidates else None


def extract_metrics(
    candidates: List[Model],
    optimizing_metrics: List[str],
) -> List[Dict[str, float]]:
    """
    A function to extract optimizing metric information for each candidate

    :param candidates: a list of sparsezoo.Model objects to find optimizing
        metric information for
    :param optimizing_metrics: a list of metric names to extract from the given
        candidates
    :returns: A list of dict objects each containing a mapping b/w optimizing metric
        name and it's value, one dict object per candidate
    """
    extracted_metrics: List[Dict[str, float]] = []
    for candidate in candidates:
        candidate_metrics = {
            metric: EXTRACTORS[metric](candidate) for metric in optimizing_metrics
        }
        extracted_metrics.append(candidate_metrics)
    return extracted_metrics


def extract_ranges(
    extracted_metrics: List[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """
    A function to extract min max metric_ranges for each optimizing metric from the
    specified extracted_metrics

    :param extracted_metrics: a list of dict objects, one for each candidate each
        containing a mapping between the metric name and it's corresponding value
        for that candidiate
    :returns: a dict object containing a mapping between metric name and a tuple
        of the form (min_value, max_value) for that metric
    """
    metric_ranges = {}
    Range = namedtuple("Range", "min, max")

    for candidate_metrics in extracted_metrics:
        for metric in candidate_metrics:
            if metric not in metric_ranges:
                metric_ranges[metric] = Range(
                    min=candidate_metrics[metric], max=candidate_metrics[metric]
                )
            else:
                old_min, old_max = metric_ranges[metric]
                metric_ranges[metric] = Range(
                    min=min(candidate_metrics[metric], old_min),
                    max=max(candidate_metrics[metric], old_max),
                )
    return metric_ranges


def get_best_model_with_metrics(
    candidates: List[Model],
    optimizing_metrics: List[str],
) -> Tuple[Optional[Model], Dict[str, float]]:
    """
    Among the possible SparseZoo candidates choose the one that is most appropriate
    given the optimizing_metrics

    :param candidates: a list of sparsezoo.Model objects representing
        candidates for finding the best model based on specified optimizing
        metrics, if no optimizing metrics are given return the first quantized
        model found
    :param optimizing_metrics: a list of strings representing different
        optimizing_metrics to prioritize when finding appropriate model
    :return: a tuple with best Model object among the candidates based on specified
        optimizing_metrics along with model_metrics
    """
    optimizing_metrics: List[str] = validate_optimizing_metrics(
        optimizing_metrics=optimizing_metrics or []
    )
    if not optimizing_metrics:
        # return first quantized model
        return first_quantized_model(candidates=candidates), {}

    filtered_candidates: List[Model] = filter_candidates(candidates, optimizing_metrics)
    if not filtered_candidates:
        raise ValueError(
            f"No filtered candidates found based on specified metrics, "
            f"{optimizing_metrics}"
        )

    extracted_metrics_for_candidates: List[Dict[str, float]] = extract_metrics(
        filtered_candidates, optimizing_metrics
    )
    metric_ranges: Dict[str, Tuple[float, float]] = extract_ranges(
        extracted_metrics_for_candidates
    )

    heuristic_for_candidates: List[float] = [
        compute_heuristics(metrics, metric_ranges)
        for metrics in extracted_metrics_for_candidates
    ]

    best_candidate_index = max(
        range(len(filtered_candidates)),
        key=lambda index: heuristic_for_candidates[index],
    )
    return (
        candidates[best_candidate_index],
        extracted_metrics_for_candidates[best_candidate_index],
    )


def compute_heuristics(
    metrics: Dict[str, float], ranges: Dict[str, Tuple[float, float]]
) -> float:
    """
    A function that returns a heuristic based on the metric values, and the
    corresponding ranges associated with each metric, the heuristic is
    calculated as the sum of min-max normalized values for all metrics

    Assuming 2 metrics m_1, m_2 with ranges min-max values as m_1_min, m_1_max
    and m_2_min, m_2_max the heuristic h will be:

    h = (m_1 - m_1_min) / (m_1_max - m_1_min) +
        (m_2 - m_2_min) / (m_2_max - m_2_min)

    :param metrics: a dict mapping between metric name and it's corresponding value
    :param ranges: a dict mapping between metric name and a tuple containing the
    metric's
        min, and max values
    :returns: a float heuristic
    """
    total = 0.0
    for metric_name, value in metrics.items():
        low, high = ranges[metric_name]
        if low == high:
            # avoid division by zero
            new_value = 0
        else:
            new_value = (value - low) / (high - low)
        assert 0.0 <= new_value <= 1.0
        # LATENCY should be: `-new_value`
        # ACCURACY should be: `new_value`
        if metric_name in MINIMIZABLE_METRICS:
            new_value = -new_value
        total += new_value
    return total


def recommend_stub(
    task: Optional[str] = None,
    dataset: Optional[str] = None,
    scenario: Optional[str] = None,
    optimizing_metrics: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[str, Dict[str, float]]:
    """
    A function that returns appropriate SparseZoo stub or deployment directory given
    the task or dataset, optimizing optimizing_metrics and a deployment scenario

    :param task: str A supported task
    :param dataset: str The public dataset this model was trained for
    :param scenario: Optional[str] target deployment scenario, ex. `vnni`
    :param optimizing_metrics: Optional[List[str] representing different
        optimizing_metrics to prioritize for when searching for models
    :return: A tuple containing best stub based on specified arguments along
        with its model metrics, i.e (stub, metrics)
    """
    domain, subdomain = infer_domain_and_subdomain(dataset=dataset, task=task)

    models: List[Model] = search_models(
        domain=domain,
        sub_domain=subdomain,
        dataset=dataset,
    )

    if not models:
        raise ValueError(
            "Could not find any relevant models for the given task, dataset "
            f": {task, dataset}"
        )

    if scenario is not None:
        models = [model for model in models if scenario.lower() in model.source]

    if not models:
        raise ValueError(
            "Could not find any relevant models for the given task, dataset "
            f"and deployment scenario: {task, dataset, scenario}"
        )

    model, metrics = get_best_model_with_metrics(
        candidates=models, optimizing_metrics=optimizing_metrics
    )
    return model.source, metrics


def infer_domain_and_subdomain(
    dataset: Optional[str],
    task: Optional[str],
) -> Tuple[str, str]:
    """
    Infer dataset, domain and subdomain from the given dataset and task. Note
    at-least one out of dataset and task must be provided. If both are specified
    task is used to determine the domain and subdomain.

    :param dataset: Optional[str] An optional dataset name, must be specified
        if task not given
    :param task: Optional[str] An optional task name, must be specified
        if task not specified
    :return: A tuple of the format (domain, subdomain)
    """
    task_info: Optional[TaskName] = get_task_info(task_name=task)
    dataset_task_info: Optional[TaskName] = get_dataset_info(dataset_name=dataset)
    if not task_info and not dataset_task_info:
        raise ValueError(
            f"Could not find any info for the given (task, dataset): {task, dataset}"
        )
    if task_info and dataset_task_info:
        if task_info.domain != dataset_task_info.domain:
            raise ValueError(
                f"Task domain: {task_info.domain} does not match dataset"
                f" domain: {dataset_task_info.domain}"
            )
        if task_info.sub_domain != dataset_task_info.sub_domain:
            raise ValueError(
                f"Task sub_domain: {task_info.sub_domain} does not match dataset"
                f" sub_domain: {dataset_task_info.sub_domain}"
            )
    domain = task_info.domain if task_info else dataset_task_info.domain
    subdomain = task_info.sub_domain if task_info else dataset_task_info.sub_domain
    return domain, subdomain
