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

import os
from datetime import datetime

import pytest

from sparsezoo.models.detection import yolo_v3
from tests.sparsezoo.helpers import model_constructor


@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "sparse_name,sparse_category,sparse_target"
    ),
    [
        (True, "pytorch", "ultralytics", "coco", None, "base", "none", None),
        (
            True,
            "pytorch",
            "ultralytics",
            "coco",
            None,
            "pruned_quant",
            "aggressive_94",
            None,
        ),
        (True, "pytorch", "ultralytics", "coco", None, "pruned", "aggressive_97", None),
    ],
)
def test_yolo_v3(
    download,
    framework,
    repo,
    dataset,
    training_scheme,
    sparse_name,
    sparse_category,
    sparse_target,
):
    model_constructor(
        constructor_function=yolo_v3,
        download=download,
        framework=framework,
        repo=repo,
        dataset=dataset,
        training_scheme=training_scheme,
        sparse_name=sparse_name,
        sparse_category=sparse_category,
        sparse_target=sparse_target,
        test_name=os.path.join(os.path.basename(__file__), str(datetime.now())),
    )
