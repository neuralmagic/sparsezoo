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

from sparsezoo.v2.objects.model import Model
from sparsezoo.v2.utils.helpers import setup_model


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95"  # noqa E501
    ],
)
class TestSetupModel:
    @pytest.fixture()
    def setup(self, stub):
        # setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        download_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Model(stub)

        yield model, temp_dir, download_dir
        temp_dir.cleanup()
        download_dir.cleanup()

    def test_setup_model_from_paths(self, setup):
        (
            model,
            temp_dir,
            download_dir,
        ) = setup

        training_path = model.training.get_path(download_directory=download_dir.name)
        deployment_path = model.deployment.get_path(
            download_directory=download_dir.name
        )
        onnx_model_path = model.onnx_model.get_path(
            download_directory=download_dir.name
        )
        sample_inputs_path = model.sample_inputs.get_path(
            download_directory=download_dir.name
        )
        recipes_path = model.recipes.get_path(download_directory=download_dir.name)

        setup_model(
            output_dir=temp_dir.name,
            training=training_path,
            deployment=deployment_path,
            onnx_model=onnx_model_path,
            sample_inputs=sample_inputs_path,
            # TODO: .get_path() needs to be supported for dict-like obj
            # sample_outputs=model.sample_outputs.get_path(),
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
        model, temp_dir, download_dir = setup
        model.download(download_dir.name)
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
