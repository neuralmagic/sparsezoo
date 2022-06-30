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
import shutil
import tempfile

import pytest

from sparsezoo import Zoo
from sparsezoo.requests.download import download_model_get_request
from sparsezoo.v2.helpers import restructure_request_json
from sparsezoo.v2.model_directory import ModelDirectory
from tests.sparsezoo.v2.test_file import _create_sample_file


@pytest.mark.parametrize(
    "stub, expected_content",
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            [
                "training",
                "sample_originals",
                "sample_inputs",
                "model.md",
                "sample_outputs",
                "sample_labels",
                "deployment",
                "onnx",
                "logs",
                "model.onnx",
                "analysis.yaml",
                "benchmarks.yaml",
                "eval.yaml",
                ["recipe_foo.md", "recipe_original.md"],
            ],
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            [
                "training",
                "sample_originals",
                "sample_inputs",
                "model.md",
                "sample_outputs",
                "sample_labels",
                "deployment",
                "onnx",
                "logs",
                "model.onnx",
                "analysis.yaml",
                "benchmarks.yaml",
                "eval.yaml",
                ["recipe_original.md", "recipe_transfer_learn.md"],
            ],
        ),
    ],
    scope="function",
)
class TestModelDirectory:
    @pytest.fixture()
    def setup(self, stub, expected_content):
        # setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Zoo.load_model_from_stub(stub)
        # directory_path = self._get_local_directory(model)
        directory_path = None

        yield directory_path, model, expected_content, temp_dir

        temp_dir.cleanup()

    def test_model_directory_from_zoo_1(self, setup):
        directory_path, model, expected_content, temp_dir = setup
        # scenario when clone_sample_outputs = True, tars = True
        request_json = self._get_api_request(
            model=model, clone_sample_outputs=True, tars=True
        )
        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        self._validate_model_directory(model_directory, expected_content, from_zoo=True)

    def test_model_directory_from_zoo_2(self, setup):
        directory_path, model, expected_content, temp_dir = setup
        # scenario when clone_sample_outputs = True, tars = False
        request_json = self._get_api_request(
            model=model, clone_sample_outputs=True, tars=False
        )
        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        self._validate_model_directory(model_directory, expected_content, from_zoo=True)

    def test_model_directory_from_zoo_3(self, setup):
        directory_path, model, expected_content, temp_dir = setup
        # scenario when clone_sample_outputs = False, tars = False
        request_json = self._get_api_request(
            model=model, clone_sample_outputs=False, tars=False
        )
        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        self._validate_model_directory(model_directory, expected_content, from_zoo=True)

    def test_model_directory_from_zoo_4(self, setup):
        directory_path, model, expected_content, temp_dir = setup
        # scenario when clone_sample_outputs = False, tars = True
        request_json = self._get_api_request(
            model=model, clone_sample_outputs=False, tars=True
        )
        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        self._validate_model_directory(model_directory, expected_content, from_zoo=True)

    # def test_model_directory_from_directory(self, setup):
    #     directory_path, request_json, expected_content, temp_dir = setup
    #
    #     model_directory = ModelDirectory.from_directory(directory_path=directory_path)
    #     self._validate_model_directory(model_directory, expected_content)

    # def test_download_with_tar_dirs(self, setup):
    #     directory_path, model, expected_content, temp_dir = setup
    #     request_json = self._get_api_request(
    #         model=model, clone_sample_outputs=False, tars=False
    #     )
    #     model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
    #     assert model_directory.download(directory_path=temp_dir.name)
    #     assert model_directory.validate()
    #     assert model_directory.validate(minimal_validation=True)
    #
    # def test_download(self, setup):
    #     directory_path, model, expected_content, temp_dir = setup
    #     request_json = self._get_api_request(
    #         model=model, clone_sample_outputs=True, tars=True
    #     )
    #     model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
    #     assert model_directory.download(directory_path=temp_dir.name)
    #     assert model_directory.validate()
    #     assert model_directory.validate(minimal_validation=True)

    # def test_validate_from_directory(self, setup):
    #     directory_path, request_json, expected_content, temp_dir = setup
    #
    #     model_directory = ModelDirectory.from_directory(directory_path=directory_path)
    #     assert model_directory.validate()
    #     assert model_directory.validate(minimal_validation=True)

    # def test_generate_outputs(self, setup):
    #     directory_path, model, expected_content, temp_dir = setup
    #     model_directory = ModelDirectory.from_directory(directory_path=directory_path)
    #     for engine in ["onnxruntime", "deepsparse"]:
    #         self._test_generate_outputs_single_engine(engine, model_directory)
    #
    # def test_analysis(self, setup):
    #     pass

    def _test_generate_outputs_single_engine(self, engine, model_directory):
        directory_path = model_directory.path
        # test whether the functionality saves the numpy files to tar properly
        tar_file_expected_path = os.path.join(
            directory_path, f"sample_outputs_{engine}.tar.gz"
        )
        if os.path.isfile(tar_file_expected_path):
            os.remove(tar_file_expected_path)

        for output_expected, output in zip(
            model_directory.sample_outputs[engine],
            model_directory.generate_outputs(engine_type=engine, save_to_tar=True),
        ):
            output_expected = list(output_expected.values())
            for o1, o2 in zip(output_expected, output):
                assert pytest.approx(o1, abs=1e-5) == o2.flatten()

        assert os.path.isfile(tar_file_expected_path)

    @staticmethod
    def _get_local_directory(model):
        model.download()
        directory_path = os.path.dirname(model.training[0].dir_path)

        # Adding several hacks to make files in
        # directory_path adhere to the desired format

        # Create framework-files
        framework_files = os.path.join(directory_path, "framework-files")
        os.rename(model.training[0].dir_path, framework_files)

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
    def _get_api_request(model, clone_sample_outputs=True, tars=False):
        request_json = download_model_get_request(args=model)["model"]["files"]
        return restructure_request_json(request_json, clone_sample_outputs, tars)

    @staticmethod
    def _validate_model_directory(model_directory, expected_content, from_zoo=False):
        for file, expected_file in zip(model_directory.files, expected_content):
            if isinstance(file, list):
                assert set([_file.name for _file in file]) == set(
                    [_expected_file for _expected_file in expected_file]
                )
            elif isinstance(file, dict):
                assert set(file.keys()).issubset(
                    {"onnxruntime", "deepsparse", "framework"}
                )
            else:
                assert expected_file == file.name
