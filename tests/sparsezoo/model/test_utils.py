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
import tempfile

import pytest

from sparsezoo.model import Model, load_files_from_stub, setup_model


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
    "deployment": {
        "model.onnx",
        "tokenizer.json",
        "config.json",
        "tokenizer_config.json",
    },
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
            "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/pruned80_quant-none-vnni",  # noqa E501
            EXPECTED_NLP_FILES,
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            EXPECTED_YOLO_FILES,
        ),
    ],
    scope="function",
)
def test_load_files_from_stub(stub, expected_files):
    if stub.startswith("zoo:"):
        stub = stub[len("zoo:") :]

    files_from_stub = load_files_from_stub(stub=stub)
    files = files_from_stub.get("files")
    model_id = files_from_stub.get("model_id")
    params = files_from_stub.get("params")
    results = files_from_stub.get("results")
    model_onnx_size_compressed_bytes = files_from_stub.get("model_onnx_size_compressed_bytes")

    for file_type, file_names_expected in expected_files.items():
        file_names = set(
            file["display_name"] for file in files if file["file_type"] == file_type
        )
        assert not file_names_expected.difference(file_names)
    assert model_id is not None
    assert params.__eq__({})
    assert results is not None
    assert model_onnx_size_compressed_bytes > 0


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/pruned80_quant-none-vnni"  # noqa E501
    ],
)
class TestSetupModel:
    @pytest.fixture()
    def setup(self, stub):
        # setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        download_dir = tempfile.TemporaryDirectory(dir="/tmp")
        yield stub, temp_dir, download_dir
        temp_dir.cleanup()
        download_dir.cleanup()

    def test_setup_model_from_paths(self, setup):
        (
            stub,
            temp_dir,
            download_dir,
        ) = setup

        model = Model(stub, download_dir.name)
        training_path = model.training.path
        deployment_path = model.deployment.path
        onnx_model_path = model.onnx_model.path
        sample_inputs_path = model.sample_inputs.path
        recipes_path = model.recipes.path

        setup_model(
            output_dir=temp_dir.name,
            training=training_path,
            deployment=deployment_path,
            onnx_model=onnx_model_path,
            sample_inputs=sample_inputs_path,
            # TODO: .path() needs to be supported for dict-like obj
            # sample_outputs=model.sample_outputs.path,
            recipes=recipes_path,
        )

        assert {
            "training",
            "deployment",
            "recipe",
            "model.onnx",
            "sample_inputs.tar.gz",
        } == set(os.listdir(temp_dir.name))

    def test_setup_model_from_objects(self, setup):
        stub, temp_dir, download_dir = setup
        model = Model(stub, download_dir.name)
        model.download()
        model.sample_inputs.unzip()

        training = model.training
        deployment = model.deployment
        onnx_model = model.onnx_model
        sample_inputs = model.sample_inputs
        # sample_outputs
        recipes = model.recipes

        setup_model(
            output_dir=temp_dir.name,
            training=training,
            deployment=deployment,
            onnx_model=onnx_model,
            sample_inputs=sample_inputs,
            recipes=recipes,
        )

        assert {
            "training",
            "deployment",
            "recipe",
            "model.onnx",
            "sample_inputs",
        } == set(os.listdir(temp_dir.name))
        download_dir.cleanup()
