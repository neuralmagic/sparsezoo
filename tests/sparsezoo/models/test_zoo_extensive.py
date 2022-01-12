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

from typing import List

import pytest
from flaky import flaky

from sparsezoo.models import Zoo
from tests.sparsezoo.helpers import download_and_verify


def _get_models(domain, sub_domain) -> List[str]:
    page = 1
    models = []
    while True:
        results = Zoo.search_models(domain, sub_domain, page=page)
        if len(results) == 0:
            break
        models.extend(results)
        page += 1
    return [model.stub for model in models]


@flaky
@pytest.mark.parametrize(("model"), _get_models("cv", "classification"))
def test_classification_models(model):
    download_and_verify(model)


@pytest.mark.parametrize(("model"), _get_models("cv", "detection"))
def test_detection_models(model):
    download_and_verify(model)
