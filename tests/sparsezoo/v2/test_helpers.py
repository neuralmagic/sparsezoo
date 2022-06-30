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

import copy

import pytest

from sparsezoo import Zoo
from sparsezoo.requests.download import download_model_get_request
from sparsezoo.v2.helpers import restructure_request_json


NLP_FILE_NAMES = {
    "deployment": {"deployment.tar.gz"},
    "training": {"training.tar.gz"},
    "outputs": {
        "sample_outputs_onnxruntime.tar.gz",
        "sample_outputs_deepsparse.tar.gz",
    },
    "onnx": {"onnx.tar.gz", "model.onnx"},
    "recipe": {"recipe_foo.md", "recipe_original.md"},
    "card": {"model.md"},
    "benchmarking": {"benchmarks.yaml", "eval.yaml", "analysis.yaml"},
    "labels": {"sample_labels.tar.gz"},
    "originals": {"sample_originals.tar.gz"},
    "inputs": {"sample_inputs.tar.gz"},
    "logs": {"logs.tar.gz"},
}
CV_FILE_NAMES = copy.copy(NLP_FILE_NAMES)
CV_FILE_NAMES["recipe"] = {"recipe_transfer_learn.md", "recipe_original.md"}
CV_FILE_NAMES["outputs"] = {"sample_outputs.tar.gz"}


@pytest.mark.parametrize(
    "model_stub, expected_file_names, clone_sample_outputs",
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            NLP_FILE_NAMES,
            True,
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            CV_FILE_NAMES,
            False,
        ),
    ],
)
def test_restructure_request_json_with_tar_dirs(
    model_stub, expected_file_names, clone_sample_outputs
):
    model = Zoo.load_model_from_stub(model_stub)
    request_json = download_model_get_request(args=model)["model"]["files"]
    request_json = restructure_request_json(
        request_json, clone_sample_outputs, tars=True
    )
    file_names = {
        (file_dict["display_name"], file_dict["file_type"])
        for file_dict in request_json
    }
    for file_type, files in expected_file_names.items():
        for display_name in files:
            file_names.remove((display_name, file_type))

    assert not file_names


NLP_TRAINING_FILES = {
    "trainer_state.json",
    "config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "pytorch_model.bin",
}
NLP_DEPLOYMENT_FILES = NLP_TRAINING_FILES | {"model.onnx"}
NLP_FILE_NAMES = {
    "deployment": NLP_DEPLOYMENT_FILES,
    "training": NLP_TRAINING_FILES,
    "outputs": {
        "sample_outputs_onnxruntime.tar.gz",
        "sample_outputs_deepsparse.tar.gz",
    },
    "onnx": {"model.onnx", "model.11.onnx", "model.14.onnx"},
    "recipe": {"recipe_foo.md", "recipe_original.md"},
    "card": {"model.md"},
    "benchmarking": {"benchmarks.yaml", "eval.yaml", "analysis.yaml"},
    "onnx_gz": {"model.onnx.tar.gz"},
    "labels": {"sample_labels.tar.gz"},
    "originals": {"sample_originals.tar.gz"},
    "inputs": {"sample_inputs.tar.gz"},
    "tar_gz": {"model.tar.gz"},
    "logs": {"logs.yaml"},
}

CV_TRAINING_FILES = {"model.pt", "model.ckpt.pt"}
CV_DEPLOYMENT_FILES = CV_TRAINING_FILES | {"model.onnx"}
CV_FILE_NAMES = {
    "deployment": CV_DEPLOYMENT_FILES,
    "training": CV_TRAINING_FILES,
    "outputs": {"sample_outputs.tar.gz"},
    "onnx": {"model.onnx", "model.11.onnx", "model.14.onnx"},
    "recipe": {"recipe_transfer_learn.md", "recipe_original.md"},
    "card": {"model.md"},
    "benchmarking": {"benchmarks.yaml", "eval.yaml", "analysis.yaml"},
    "onnx_gz": {"model.onnx.tar.gz"},
    "labels": {"sample_labels.tar.gz"},
    "originals": {"sample_originals.tar.gz"},
    "inputs": {"sample_inputs.tar.gz"},
    "tar_gz": {"model.tar.gz"},
    "logs": {"logs.yaml"},
}


@pytest.mark.parametrize(
    "model_stub, expected_file_names, clone_sample_outputs",
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            NLP_FILE_NAMES,
            True,
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            CV_FILE_NAMES,
            False,
        ),
    ],
)
def test_restructure_request_json(
    model_stub, expected_file_names, clone_sample_outputs
):
    model = Zoo.load_model_from_stub(model_stub)
    request_json = download_model_get_request(args=model)["model"]["files"]
    request_json = restructure_request_json(
        request_json, clone_sample_outputs, tars=False
    )
    file_names = {
        (file_dict["display_name"], file_dict["file_type"])
        for file_dict in request_json
    }
    for file_type, files in expected_file_names.items():
        for display_name in files:
            file_names.remove((display_name, file_type))

    assert not file_names
