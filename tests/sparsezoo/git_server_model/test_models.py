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
pytest /Users/george/neuralmagic/sparsezoo/tests/sparsezoo/git_server_model/test_models.py


"""
from sparsezoo.git_server_model.models.modelcard_metadata import (
    GitServerModelCardMetadata,
)
from sparsezoo.git_server_model.utils import (
    extract_git_server_metadata,
    local_load,
    web_load,
)


def test_model_card_local_load():
    """
    Test local_load()
    """
    # Your modelcard path
    path = "/Users/george/Desktop/model.md"
    metadata = local_load(path)
    output = {
        "card_version": "0.8.0",
        "domain": "cv",
        "sub_domain": "classification",
        "architecture": "resnet_v1",
        "sub_architecture": 50,
        "framework": "pytorch",
        "repo": "sparseml",
        "version": None,
        "dataset": "imagenet",
        "training_scheme": None,
        "sparse_tag": "pruned95-none",
        "display_name": "95% Pruned ResNet-50",
        "display_description": "Pruned ResNet-50 architecture trained on the ImageNet dataset. 75.9% top1 validation accuracy, recovering over 99% of the top1 validation accuracy of the baseline model (76.1%).",
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
    }
    assert metadata == output, "[test_model_card_local_load]: local_load(path) failed"


def test_git_server_metadata():
    paths = [
        "https://git.neuralmagic.com/neuralmagic/cary",
        "git@git.neuralmagic.com:neuralmagic/cary.git",
        "https://git.neuralmagic.com/neuralmagic/cary.git",
    ]

    for path in paths:
        meta = extract_git_server_metadata(path)
        assert meta["name"] == "cary"
        assert meta["namespace"] == "neuralmagic"


def test_web_load():
    """
    test web_load()
    """
    paths = [
        "https://git.neuralmagic.com/neuralmagic/cary",
        "git@git.neuralmagic.com:neuralmagic/cary.git",
        "https://git.neuralmagic.com/neuralmagic/cary.git",
    ]
    filename = "model.md"
    for path in paths:
        payload = web_load(git_server_url=path, filename=filename)


# Test metadata popuilation using class
