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
import os
import shutil
import tempfile

import pytest

from sparsezoo import Zoo
from sparsezoo.requests.download import download_model_get_request
from sparsezoo.v2.model_directory import ModelDirectory
from tests.sparsezoo.v2.test_file import _create_sample_file


@pytest.mark.parametrize(
    "domain, sub_domain, model_index, expected_content",
    [
        (
            "cv",
            "classification",
            0,
            [
                "framework-files",
                "sample-originals",
                "sample-inputs",
                "sample-outputs",
                "sample-labels",
                "model.onnx",
                ["model.1.onnx", "model.2.onnx"],
                "analysis.yaml",
                "benchmarks.yaml",
                "eval.yaml",
                "model.md",
                ["recipe_foo.md", "recipe_bar.md"],
            ],
        ),
        (
            "nlp",
            "question_answering",
            0,
            [
                "framework-files",
                "sample-originals",
                "sample-inputs",
                "sample-outputs",
                "sample-labels",
                "model.onnx",
                ["model.1.onnx", "model.2.onnx"],
                "analysis.yaml",
                "benchmarks.yaml",
                "eval.yaml",
                "model.md",
                ["recipe_foo.md", "recipe_bar.md"],
            ],
        ),
    ],
    scope="function",
)
class TestModelDirectory:
    @pytest.fixture()
    def setup(self, domain, sub_domain, model_index, expected_content):
        # setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Zoo.search_models(
            domain=domain, sub_domain=sub_domain, override_folder_name=temp_dir.name
        )[model_index]
        request_json = self._get_api_request(model)
        directory_path = self._get_local_directory(model)

        yield directory_path, request_json, expected_content, temp_dir

        temp_dir.cleanup()

    def test_model_directory_from_zoo(self, setup):
        directory_path, request_json, expected_content, temp_dir = setup

        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        self._validate_model_directory(model_directory, expected_content, from_zoo=True)

    def test_model_directory_from_directory(self, setup):
        directory_path, request_json, expected_content, temp_dir = setup

        model_directory = ModelDirectory.from_directory(directory_path=directory_path)
        self._validate_model_directory(model_directory, expected_content)

    def test_download(self, setup):
        directory_path, request_json, expected_content, temp_dir = setup

        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        assert model_directory.download(directory_path=temp_dir.name)

    def test_validate_from_directory(self, setup):
        directory_path, request_json, expected_content, temp_dir = setup

        model_directory = ModelDirectory.from_directory(directory_path=directory_path)
        assert model_directory.validate()

    def test_validate_from_zoo_api(self, setup):
        _, request_json, expected_content, temp_dir = setup

        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        # TODO: For now, skipping this test. We
        # need proper model located in the NeuralMagic' server.
        # assert model_directory.validate()
        assert model_directory

    def test_generate_outputs_onnxruntime(self, setup):
        directory_path, request_json, expected_content, temp_dir = setup

        model_directory = ModelDirectory.from_directory(directory_path=directory_path)
        for output_expected, output in zip(
            model_directory.sample_outputs,
            model_directory.generate_outputs(engine_type="onnxruntime"),
        ):
            output_expected = list(output_expected.values())
            for o1, o2 in zip(output_expected, output):
                pytest.approx(o1, abs=1e-5) == o2.flatten()

    def test_generate_outputs_deepsparse(self, setup):
        directory_path, request_json, expected_content, temp_dir = setup
        model_directory = ModelDirectory.from_directory(directory_path=directory_path)
        for output_expected, output in zip(
            model_directory.sample_outputs,
            model_directory.generate_outputs(engine_type="deepsparse"),
        ):
            output_expected = list(output_expected.values())
            for o1, o2 in zip(output_expected, output):
                pytest.approx(o1, abs=1e-5) == o2.flatten()

    def test_analysis(self, setup):
        pass

    @staticmethod
    def _get_local_directory(model):
        model.download()
        directory_path = os.path.dirname(model.framework_files[0].dir_path)

        # Adding several hacks to make files in
        # directory_path adhere to the desired format

        # Create framework-files
        framework_files = os.path.join(directory_path, "framework-files")
        os.rename(model.framework_files[0].dir_path, framework_files)

        # Create onnx_models(opsets)
        for opset in range(1, 3):
            _create_sample_file(os.path.join(directory_path, f"model.{opset}.onnx"))

        # Create yaml and md files
        for name in [
            "analysis.yaml",
            "benchmarks.yaml",
            "eval.yaml",
            "recipe_foo.md",
            "recipe_bar.md",
        ]:
            _create_sample_file(os.path.join(directory_path, name))

            # Create sample labels (if not there, e.g. for transformers)
            shutil.copyfile(
                os.path.join(directory_path, "sample-inputs.tar.gz"),
                os.path.join(directory_path, "sample-labels.tar.gz"),
            )

            # Create sample originals (if not there, e.g. for transformers)
            shutil.copyfile(
                os.path.join(directory_path, "sample-inputs.tar.gz"),
                os.path.join(directory_path, "sample-originals.tar.gz"),
            )

        return directory_path

    @staticmethod
    def _get_api_request(model):
        request_json = download_model_get_request(args=model)["model"]["files"]

        # Adding several hacks to make request_json adhere to the desired format

        # Create framework-files
        framework_files = [x for x in request_json if x["file_type"] == "framework"]
        request_json.append(
            {"display_name": "framework-files", "contents": framework_files}
        )
        [request_json.remove(file) for file in framework_files]

        # Create onnx_models (opsets)
        onnx_model_file = [
            x for x in request_json if x["display_name"] == "model.onnx"
        ][0]
        for opset in range(1, 3):
            _onnx_model_file = copy.deepcopy(onnx_model_file)
            _onnx_model_file["display_name"] = f"model.{opset}.onnx"
            request_json.append(_onnx_model_file)

        # Create yaml and md files
        for name in [
            "analysis.yaml",
            "benchmarks.yaml",
            "eval.yaml",
            "recipe_foo.md",
            "recipe_bar.md",
        ]:
            request_json.append({"display_name": name})

        # Create sample labels (if not there, e.g. for transformers)
        if not any(
            [x for x in request_json if x["display_name"] == "sample-labels.tar.gz"]
        ):
            request_json.append({"display_name": "sample-labels.tar.gz"})

        # Create sample originals (if not there, e.g. for transformers)
        if not any(
            [x for x in request_json if x["display_name"] == "sample-originals.tar.gz"]
        ):
            sample_originals = copy.deepcopy(
                [
                    x
                    for x in request_json
                    if x["display_name"] == "sample-inputs.tar.gz"
                ][0]
            )
            sample_originals["display_name"] = "sample-originals.tar.gz"
            request_json.append(sample_originals)

        return request_json

    @staticmethod
    def _validate_model_directory(model_directory, expected_content, from_zoo=False):
        for file, expected_file in zip(model_directory.files, expected_content):
            if isinstance(file, list):
                assert set([_file.name for _file in file]) == set(
                    [_expected_file for _expected_file in expected_file]
                )
            elif isinstance(file, dict):
                pass
            else:
                if from_zoo and file.name.endswith(".tar.gz"):
                    expected_file += ".tar.gz"
                assert expected_file == file.name
