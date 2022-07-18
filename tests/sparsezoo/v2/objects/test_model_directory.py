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
from pathlib import Path

import pytest

from sparsezoo.v2.objects.model_directory import ModelDirectory
from tests.sparsezoo.v2.objects.test_model_directory_download import (
    TestModelDirectoryFromZooApi,
)


@pytest.mark.parametrize(
    "stub, clone_sample_outputs",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
            True,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            False,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            True,
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            True,
        ),
    ],
    scope="function",
)
class TestModelDirectory:
    @pytest.fixture()
    def setup(self, stub, clone_sample_outputs):
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        request_json = TestModelDirectoryFromZooApi._get_request_json(stub)
        if clone_sample_outputs:
            request_json = TestModelDirectoryFromZooApi._clone_sample_outputs(
                request_json
            )
        model_directory = ModelDirectory.from_zoo_api(request_json)
        model_directory.download(directory_path=temp_dir.name)
        self._add_mock_files(temp_dir.name)
        model_directory = ModelDirectory.from_directory(temp_dir.name)

        yield model_directory, clone_sample_outputs

        shutil.rmtree(temp_dir.name)

    def test_validate(self, setup):
        model_directory, clone_sample_outputs = setup
        # TODO: Resolve this problem here
        assert model_directory.validate(validate_onnxruntime=clone_sample_outputs)
        assert model_directory.validate(
            validate_onnxruntime=clone_sample_outputs, minimal_validation=True
        )

    def test_generate_outputs(self, setup):
        model_directory, clone_sample_outputs = setup
        if clone_sample_outputs:
            for engine in ["onnxruntime"]:
                self._test_generate_outputs_single_engine(engine, model_directory)

    def test_analysis(self, setup):
        pass

    @staticmethod
    def _add_mock_files(directory_path: str):
        # add some mock files, to complete the full set of
        # possible expected files in the `ModelDirectory`
        # class object

        # add onnx directory
        onnx_folder_dir = os.path.join(directory_path, "onnx")
        os.makedirs(onnx_folder_dir)
        for opset in range(1, 3):
            shutil.copyfile(
                os.path.join(directory_path, "model.onnx"),
                os.path.join(onnx_folder_dir, f"model.{opset}.onnx"),
            )

        # add logs directory
        logs_folder_dir = os.path.join(directory_path, "logs")
        os.makedirs(logs_folder_dir)

        # fix yaml files
        # (they are originally constructed from .md files
        # and thus fail when being validated)
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".yaml"):
                file_dir = os.path.join(directory_path, file_name)
                os.remove(file_dir)
                Path(file_dir).touch()

        # special case for `recipe.yaml` in transformers' tests
        optional_recipe_yaml = os.path.join(directory_path, "training", "recipe.yaml")
        if os.path.isfile(optional_recipe_yaml):
            os.remove(optional_recipe_yaml)
            Path(optional_recipe_yaml).touch()

    def _test_generate_outputs_single_engine(self, engine, model_directory):
        directory_path = model_directory.path
        save_to_tar = False
        if engine == "onnxruntime":
            # test whether the functionality saves the numpy files to tar properly
            tar_file_expected_path = os.path.join(
                directory_path, f"sample_outputs_{engine}.tar.gz"
            )
            if os.path.isfile(tar_file_expected_path):
                os.remove(tar_file_expected_path)
            save_to_tar = True

        output_expected = next(iter(model_directory.sample_outputs[engine]))
        output_expected = list(output_expected.values())
        output = next(
            iter(
                model_directory.generate_outputs(
                    engine_type=engine, save_to_tar=save_to_tar
                )
            )
        )

        for o1, o2 in zip(output_expected, output):
            if o1.ndim != o2.ndim:
                o2 = o2.squeeze(0)
            # for 'onnxruntime' accuracy, we use the default one (1e-5).
            # for 'deepsparse' accuracy, we need to be mindful that we are
            # comparing deepsparse inference output with onnxruntime gt output.
            # this is why we lower the accuracy here (1e-4)
            if engine == "onnxruntime":
                assert pytest.approx(o1, abs=1e-5) == o2
            else:
                assert pytest.approx(o1, abs=1e-4) == o2

        if engine == "onnxruntime":
            assert os.path.isfile(tar_file_expected_path)
