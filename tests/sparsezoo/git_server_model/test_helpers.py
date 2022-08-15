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

import pytest

from sparsezoo.git_server.utils import extract_git_server_metadata, local_load, web_load


FIXTURE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fixtures")


@pytest.mark.parametrize(
    "path",
    [
        "https://git.neuralmagic.com/neuralmagic/cary",
        "git@git.neuralmagic.com:neuralmagic/cary.git",
        "https://git.neuralmagic.com/neuralmagic/cary.git",
    ],
    scope="function",
)
def test_git_server_metadata(path):

    meta = extract_git_server_metadata(path)
    assert meta["name"] == "cary"
    assert meta["namespace"] == "neuralmagic"


@pytest.mark.parametrize(
    "path",
    [
        FIXTURE_PATH,
    ],
    scope="function",
)
def test_model_card_local_load(path):

    metadata = local_load(path, file_name="model.md")
    expected = {
        "card_version": "1.0.0",
        "domain": "cv",
        "base": "_some_base_stub",
        "task": "classification",
        "architecture": "resnet_v1",
        "sub_architecture": 50,
        "framework": "pytorch",
        "repo": "sparseml",
        "version": None,
        "source_dataset": "imagenet",
        "train_dataset": "imagenet_2",
        "display_name": "95% Pruned ResNet-50",
        "tags": [
            "resnet",
            "resnet_v1",
            "resnet50",
            "pruned",
            "pruned95",
            "sparseml",
            "pytorch",
            "imagenet",
        ],
        "parent": "_some_parent_stub",
        "optimizations": ["GMP 95%", "QAT Int8"],
        "commands": {
            "deploy": {
                "command1": "sparseml.command.dosomething stub",
                "deploy_model": "deepsparse.command.dosomething stub",
            },
            "train": {
                "command3": "sparseml.command.foo",
                "train_model": "sparseml.command.bar stub",
                "train_model_stop_at_epoch_20": "python3 sparseml.command.bar",
            },
            "benchmark": {
                "benchmark": "deepsparse.benchmark stub",
                "benchmark_on_instance": (
                    "deepsparse.benchmark --instance_type c5.12xlarge --stub zoo_stub"
                ),
            },
        },
    }
    assert expected == metadata, "[test_model_card_local_load]: local_load(path) failed"


@pytest.mark.parametrize(
    "path",
    [
        "https://git.neuralmagic.com/neuralmagic/cary",
        "git@git.neuralmagic.com:neuralmagic/cary.git",
        "https://git.neuralmagic.com/neuralmagic/cary.git",
    ],
    scope="function",
)
def test_web_load(path):

    payload = web_load(git_server_url=path, file_name="model.md")
    expected = {
        "card_version": "1.0.0",
        "domain": "cv",
        "base": "_some_base_stub",
        "task": "classification",
        "architecture": "resnet_v1",
        "sub_architecture": 50,
        "framework": "pytorch",
        "repo": "sparseml",
        "version": None,
        "source_dataset": "imagenet",
        "train_dataset": "imagenet_2",
        "display_name": "95% Pruned ResNet-50",
        "tags": [
            "resnet",
            "resnet_v1",
            "resnet50",
            "pruned",
            "pruned95",
            "sparseml",
            "pytorch",
            "imagenet",
        ],
        "parent": "_some_parent_stub",
        "optimizations": ["GMP 95%", "QAT Int8"],
        "commands": {
            "deploy": {
                "command1": "sparseml.command.dosomething stub",
                "deploy_model": "deepsparse.command.dosomething stub",
            },
            "train": {
                "command3": "sparseml.command.foo",
                "train_model": "sparseml.command.bar stub",
                "train_model_stop_at_epoch_20": "python3 sparseml.command.bar",
            },
            "benchmark": {
                "benchmark": "deepsparse.benchmark stub",
                "benchmark_on_instance": (
                    "deepsparse.benchmark --instance_type c5.12xlarge --stub zoo_stub"
                ),
            },
        },
    }
    assert expected == payload, "[test_model_card_local_load]: local_load(path) failed"
