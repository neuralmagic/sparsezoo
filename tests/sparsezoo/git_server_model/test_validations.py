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

from sparsezoo.git_server import Benchmark, ModelCardMetadata


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
def test_benchmark(path):
    benchmark = Benchmark(path, platform="web")

    # check if there are values, metadata may update, so will not check values
    assert (
        benchmark.benchmarks
        and benchmark.git_ssh_url
        and benchmark.deepsparse_version
        and benchmark.model_commit_sha
        and benchmark.results
    ), "Benchmark metadata not populated correctly"


@pytest.mark.parametrize(
    "path, expected",
    [
        (
            FIXTURE_PATH,
            {
                "card_version": "1.0.0",
                "base": "_some_base_stub",
                "parent": None,
                "domain": "cv",
                "task": "classification",
                "architecture": "resnet_v1",
                "framework": "pytorch",
                "repo": "sparseml",
                "source_dataset": "imagenet",
                "train_dataset": "imagenet_2",
                "optimizations": "pruned95-none",
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
                "commands": None,
            },
        )
    ],
    scope="function",
)
def test_model_card_metadata(path, expected):
    model_metadata = ModelCardMetadata(path=path, platform="local")

    # model card should be immutable
    assert model_metadata.card_version == expected["card_version"], (
        "card version mismatch. "
        f"Expected {expected['card_version']}, got {model_metadata.card_version}"
    )
    assert expected["base"] == model_metadata.base, (
        "base mismatch. " f"Expected {expected['base']}, got {model_metadata.base}"
    )
    assert expected["parent"] == model_metadata.parent, (
        "parent mismatch. "
        f"Expected {expected['parent']}, got {model_metadata.parent}"
    )
    assert expected["domain"] == model_metadata.domain, (
        "domain mismatch. "
        f"Expected {expected['domain']}, got {model_metadata.domain}"
    )
    assert expected["task"] == model_metadata.task, (
        "task mismatch. " f"Expected {expected['task']}, got {model_metadata.task}"
    )
    assert expected["architecture"] == model_metadata.architecture, (
        "architecture mismatch. "
        f"Expected {expected['architecture']}, got {model_metadata.architecture}"
    )
    assert expected["framework"] == model_metadata.framework, (
        "framework mismatch. "
        f"Expected {expected['framework']}, got {model_metadata.framework}"
    )
    assert expected["repo"] == model_metadata.repo, (
        "repo mismatch. " f"Expected {expected['repo']}, got {model_metadata.repo}"
    )
    assert expected["source_dataset"] == model_metadata.source_dataset, (
        "source_dataset mismatch. "
        f"Expected {expected['source_dataset']}, got {model_metadata.source_dataset}"
    )
    assert expected["train_dataset"] == model_metadata.train_dataset, (
        "train_dataset mismatch. "
        f"Expected {expected['train_dataset']}, got {model_metadata.train_dataset}"
    )
    assert expected["optimizations"] == model_metadata.optimizations, (
        "optimizations mismatch. "
        f"Expected {expected['optimizations']}, got {model_metadata.optimizations}"
    )
    assert expected["display_name"] == model_metadata.display_name, (
        "display_name mismatch. "
        f"Expected {expected['display_name']}, got {model_metadata.display_name}"
    )
    assert expected["tags"] == model_metadata.tags, (
        "tags mismatch. " f"Expected {expected['tags']}, got {model_metadata.tags}"
    )
    assert expected["commands"] == model_metadata.commands, (
        "commands mismatch. "
        f"Expected {expected['commands']}, got {model_metadata.commands}"
    )
