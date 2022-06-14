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

import glob
import os
import tempfile

import pytest

from sparsezoo import Zoo
from sparsezoo.v2 import Directory, File
from sparsezoo.v2.helpers import setup_model_directory


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95"  # noqa E501
    ],
)
class TestSetupModelDirectory:
    @pytest.fixture()
    def setup(self, stub):
        # setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Zoo.download_model_from_stub(stub, override_folder_name=temp_dir.name)

        yield model, temp_dir
        temp_dir.cleanup()

    def test_setup_model_directory_from_paths(self, setup):
        (
            model,
            temp_dir,
        ) = setup
        output_dir = tempfile.TemporaryDirectory(dir="/tmp")

        setup_model_directory(
            output_dir=output_dir.name,
            training=model.framework_files[0].dir_path,
            deployment=model.framework_files[0].dir_path,
            onnx_model=model.onnx_file.path,
            sample_inputs=model.data_inputs.path,
            sample_outputs=model.data_outputs.path,
            recipes=[model.recipes[0].path] * 2,
        )

        folders = glob.glob(os.path.join(output_dir.name, "*"))
        assert {
            "training",
            "deployment",
            "original.md",
            "model.onnx",
            "sample-outputs.tar.gz",
            "sample-inputs.tar.gz",
        } == set(os.path.basename(file) for file in folders)
        output_dir.cleanup()

    def test_setup_model_directory(self, setup):
        model, temp_dir = setup
        output_dir = tempfile.TemporaryDirectory(dir="/tmp")

        training_folder_path = model.framework_files[0].dir_path
        training = File(
            name=os.path.basename(training_folder_path), path=training_folder_path
        )
        training = Directory.from_file(file=training)

        deployment_folder_path = model.framework_files[0].dir_path
        deployment = File(
            name=os.path.basename(deployment_folder_path), path=deployment_folder_path
        )
        deployment = Directory.from_file(file=deployment)

        onnx_model = File(
            name=os.path.basename(model.onnx_file.path), path=model.onnx_file.path
        )

        sample_inputs = File(
            name=os.path.basename(model.data_inputs.path), path=model.data_inputs.path
        )

        sample_outputs = File(
            name=os.path.basename(model.data_outputs.path), path=model.data_outputs.path
        )

        recipes = [
            File(
                name=os.path.basename(model.recipes[0].path), path=model.recipes[0].path
            )
        ] * 2

        setup_model_directory(
            output_dir=output_dir.name,
            training=training,
            deployment=deployment,
            onnx_model=onnx_model,
            sample_inputs=sample_inputs,
            sample_outputs=sample_outputs,
            recipes=recipes,
        )
        folders = glob.glob(os.path.join(output_dir.name, "*"))
        assert {
            "training",
            "deployment",
            "original.md",
            "model.onnx",
            "sample-outputs.tar.gz",
            "sample-inputs.tar.gz",
        } == set(os.path.basename(file) for file in folders)
        output_dir.cleanup()
