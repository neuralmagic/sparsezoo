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

from sparsezoo.v2.utils.backwards_compatibility import restructure_request_json
from sparsezoo.v2.requests.requests import download_get_request
BASE_API_URL = (
    os.getenv("SPARSEZOO_API_URL")
    if os.getenv("SPARSEZOO_API_URL")
    else "https://api.neuralmagic.com"
)
MODELS_API_URL = f"{BASE_API_URL}/models"


EXPECTED_IC_FILES = {
    "originals": {"sample_originals.tar.gz"},
    "outputs": {"sample_outputs.tar.gz"},
    "recipe": {"recipe_original.md", "recipe_transfer_learn.md"},
    "labels": {"sample_labels.tar.gz"},
    "onnx": {"model.onnx"},
    "training": {"model.pth"},
    "card": {"model.md"},
    "inputs": {"sample_inputs.tar.gz"},
    "deployment": {"model.onnx"},
}
EXPECTED_NLP_FILES = {
    "outputs": {"sample_outputs.tar.gz"},
    "recipe": {"recipe_original.md"},
    "onnx": {"model.onnx"},
    "training": {
        "pytorch_model.bin",
        "training_args.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json",
        "config.json",
    },
    "card": {"model.md"},
    "inputs": {"sample_inputs.tar.gz"},
    "deployment": {"model.onnx", "tokenizer.json", "config.json"},
}
EXPECTED_YOLO_FILES = {
    "originals": {"sample_originals.tar.gz"},
    "outputs": {"sample_outputs.tar.gz"},
    "recipe": {"recipe_original.md", "recipe_transfer_learn.md"},
    "onnx": {"model.onnx"},
    "training": {"model.ckpt.pt", "model.pt"},
    "card": {"model.md"},
    "inputs": {"sample_inputs.tar.gz"},
    "deployment": {"model.onnx"},
}


@pytest.mark.parametrize(
    "stub, expected_files",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
            EXPECTED_IC_FILES,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            EXPECTED_NLP_FILES,
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            EXPECTED_YOLO_FILES,
        ),
    ],
    scope="function",
)
def test_restructure_request_json(stub, expected_files):
    if stub.startswith("zoo:"):
        stub = stub[len("zoo:") :]
    request_json = download_get_request(base_url = MODELS_API_URL, args=stub)
    request_json = restructure_request_json(request_json["model"]["files"])
    for file_type, file_names_expected in expected_files.items():
        file_names = set(
            file["display_name"]
            for file in request_json
            if file["file_type"] == file_type
        )
        assert not file_names_expected.difference(file_names)
