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


__all__ = [
    "deployment_package",
]

from typing import Any, Iterable, Mapping, Optional, Union

from sparsezoo.analytics import sparsezoo_analytics
from sparsezoo.deployment_package.utils.utils import recommend_stub


@sparsezoo_analytics.send_event_decorator("cli__package")
def deployment_package(
    task: Optional[str] = None,
    dataset: Optional[str] = None,
    scenario: Optional[str] = None,
    optimizing_metric: Optional[Union[Iterable[str], str]] = None,
    **kwargs,
) -> Mapping[str, Any]:
    """
    A function that returns appropriate SparseZoo stub or deployment directory given
    the task or dataset, optimizing criterions and a deployment scenario

    :param task: str A supported task
    :param dataset: str The public dataset this model was trained for
    :param scenario: Optional[str] `VNNI` or `vnni for a VNNI compatible machine
    :param optimizing_metric: an optional string or list of strings
        representing different metrics to prioritize for when searching for models
    :return: A dict type object with the relevant stub and model metrics
    """
    optimizing_metric = (
        [optimizing_metric] if isinstance(optimizing_metric, str) else optimizing_metric
    )

    stub, metrics = recommend_stub(
        task=task,
        dataset=dataset,
        optimizing_metrics=optimizing_metric,
        scenario=scenario,
    )

    return {
        "stub": stub,
        "metrics": metrics,
    }
