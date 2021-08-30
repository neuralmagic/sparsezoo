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

from sparsezoo.models.classification import (
    resnet_18,
    resnet_34,
    resnet_50,
    resnet_50_2x,
    resnet_101,
    resnet_101_2x,
    resnet_152,
)
from tests.sparsezoo.helpers import model_constructor


@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "sparse_name,sparse_category,sparse_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_18(
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
        resnet_18,
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
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_34(
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
        resnet_34,
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
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
    ],
)
def test_resnet_50(
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
        resnet_50,
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
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
        (
            True,
            "pytorch",
            "sparseml",
            "imagenet-augmented",
            None,
            "pruned_quant",
            "aggressive",
            None,
        ),
        (True, "tensorflow_v1", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenette", None, "base", "none", None),
        (
            True,
            "pytorch",
            "sparseml",
            "imagenette",
            None,
            "pruned",
            "conservative",
            None,
        ),
        (
            True,
            "pytorch",
            "torchvision",
            "imagenette",
            None,
            "pruned",
            "conservative",
            None,
        ),
    ],
)
def test_resnet_50_extended(
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
        resnet_50,
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
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_50_2x(
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
        resnet_50_2x,
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
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_101(
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
        resnet_101,
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
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_101_2x(
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
        resnet_101_2x,
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
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_152(
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
        resnet_152,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        sparse_name,
        sparse_category,
        sparse_target,
    )
