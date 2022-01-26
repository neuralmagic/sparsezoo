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

from sparsezoo.models.classification import (
    vgg_11,
    vgg_11bn,
    vgg_13,
    vgg_13bn,
    vgg_16,
    vgg_16bn,
    vgg_19,
    vgg_19bn,
)
from tests.sparsezoo.helpers import model_constructor


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
def test_vgg_11(
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
        constructor_function=vgg_11,
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
def test_vgg_11_bn(
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
        constructor_function=vgg_11bn,
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
def test_vgg_13(
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
        constructor_function=vgg_13,
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
def test_vgg_13_bn(
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
        constructor_function=vgg_13bn,
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


@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "sparse_name,sparse_category,sparse_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_vgg_16(
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
        constructor_function=vgg_16,
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
def test_vgg_16_bn(
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
        constructor_function=vgg_16bn,
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
def test_vgg_19(
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
        constructor_function=vgg_19,
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
def test_vgg_19_bn(
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
        constructor_function=vgg_19bn,
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
