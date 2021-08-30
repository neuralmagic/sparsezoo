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

from sparsezoo.models.detection import ssd_resnet50_300
from tests.sparsezoo.helpers import model_constructor


@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "sparse_name,sparse_category,sparse_target"
    ),
    [
        (True, "pytorch", "sparseml", "coco", None, "base", "none", None),
        (True, "pytorch", "sparseml", "coco", None, "pruned", "moderate", None),
    ],
)
def test_ssd_resnet50_300(
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
        ssd_resnet50_300,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        sparse_name,
        sparse_category,
        sparse_target,
    )


@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "sparse_name,sparse_category,sparse_target"
    ),
    [
        (True, "pytorch", "sparseml", "voc", None, "base", "none", None),
        (True, "pytorch", "sparseml", "voc", None, "pruned", "moderate", None),
    ],
)
def test_ssd_resnet50_300_extended(
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
        ssd_resnet50_300,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        sparse_name,
        sparse_category,
        sparse_target,
    )
