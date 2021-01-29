# Copyright 2021-present Neuralmagic, Inc.
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
import shutil

import pytest

from sparsezoo.models.classification import mobilenet_v1
from sparsezoo.utils import CACHE_DIR
from tests.sparsezoo.utils import validate_downloaded_model


@pytest.mark.parametrize(
    "download,framework,repo,dataset,training_scheme,"
    "optim_name,optim_category,optim_target",
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
    ],
)
def test_constructor(
    download,
    framework,
    repo,
    dataset,
    training_scheme,
    optim_name,
    optim_category,
    optim_target,
):
    other_args = {
        "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
    }

    if framework is None:
        model = mobilenet_v1(**other_args)
    else:
        model = mobilenet_v1(
            framework,
            repo,
            dataset,
            training_scheme,
            optim_name,
            optim_category,
            optim_target,
            **other_args,
        )
    assert model

    if download:
        model.download(overwrite=True)
        validate_downloaded_model(model, check_other_args=other_args)
        shutil.rmtree(model.dir_path)
