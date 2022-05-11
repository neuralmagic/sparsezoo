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

from src.sparsezoo import Zoo
from src.sparsezoo.objects.model_object import ModelDirectory
from src.sparsezoo.requests.download import download_model_get_request


@pytest.mark.parametrize(
    "domain, sub_domain, model_index, expected_content",
    [
        (
            "cv",
            "classification",
            0,
            {
                "framework_files": "framework-files",
                "sample_originals": "sample-originals.tar.gz",
                "sample_inputs": "sample-inputs.tar.gz",
                "sample_outputs": "sample-outputs.tar.gz",
                "sample_labels": "sample-labels.tar.gz",
                "onnx_model": "model.onnx",
                "onnx_models": None,  # there are no additional onnx models
                "analysis": None,  # there is no analysis file
                "benchmarks": None,  # there is no benchmarks file
                "eval_results": None,  # there is no eval file
                "model_card": "model.md",
                "recipes": None,  # there are recipes in proper format
            },
        ),
        (
            "nlp",
            "question_answering",
            0,
            {
                "framework_files": "framework-files",
                "sample_originals": None,  # there are no sample originals
                "sample_inputs": "sample-inputs.tar.gz",
                "sample_outputs": "sample-outputs.tar.gz",
                "sample_labels": None,  # there are no sample labels
                "onnx_model": "model.onnx",
                "onnx_models": None,  # there are no additional onnx models
                "analysis": None,  # there is no analysis file
                "benchmarks": None,  # there is no benchmarks file
                "eval_results": None,  # there is no eval file
                "model_card": "model.md",
                "recipes": None,  # there are recipes in proper format
            },
        ),
    ],
    scope="function",
)
class TestFromSparseZoo:
    @pytest.fixture()
    def setup(self, domain, sub_domain, model_index, expected_content):
        # setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Zoo.search_models(
            domain=domain, sub_domain=sub_domain, override_folder_name=temp_dir.name
        )[model_index]
        model.download()
        model_directory = os.path.dirname(model.framework_files[0].dir_path)

        yield model, expected_content, model_directory

        temp_dir.cleanup()

    def test_from_sparsezoo(self, setup):
        model, expected_content, _ = setup

        request_json = self._get_api_request(model)
        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        self._validate_model_directory(model_directory, expected_content)

    def test_from_local_directory(self, setup):
        _, expected_content, directory_path = setup

        model_directory = ModelDirectory.from_directory(directory_path=directory_path)
        self._validate_model_directory(model_directory, expected_content)

    @staticmethod
    def _get_api_request(model):
        request_json = download_model_get_request(args=model)["model"]
        return request_json

    @staticmethod
    def _validate_model_directory(model_directory, expected_content):
        for attribute, file in vars(model_directory).items():
            expected_file = expected_content[attribute]
            if file is None:
                assert expected_file is None
            else:
                assert expected_file == file.name
