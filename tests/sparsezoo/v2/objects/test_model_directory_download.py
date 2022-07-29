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
from sparsezoo.requests import download_model_get_request
from sparsezoo.v2.objects.model_directory import ModelDirectory
from sparsezoo.v2.utils.backwards_compatibility import (
    fetch_from_request_json,
    restructure_request_json,
)


files_ic = {
    "analysis.yaml",
    "deployment.tar.gz",
    "logs.tar.gz",
    "model.onnx",
    "recipe_original.md",
    "sample_inputs.tar.gz",
    "sample_originals.tar.gz",
    "benchmarks.yaml",
    "eval.yaml",
    "model.md",
    "onnx.tar.gz",
    "recipe_transfer_learn.md",
    "sample_labels.tar.gz",
    "sample_outputs.tar.gz",
    "training.tar.gz",
}

files_nlp = copy.copy(files_ic)
files_nlp.remove("recipe_transfer_learn.md")
files_yolo = copy.copy(files_ic)


@pytest.mark.parametrize(
    "stub, expected_files",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
            files_ic,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            files_nlp,
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            files_yolo,
        ),
    ],
    scope="function",
)
class TestModelDirectoryFromZooApi:
    @pytest.fixture()
    def setup(self, stub, expected_files):
        # setup
        request_json = self._get_request_json(stub)

        yield request_json, expected_files

    def test_model_directory_from_zoo_1(self, setup):
        self._test_model_directory_from_zoo(
            setup, download_folders_as_tars=True, multiple_sample_outputs=True
        )

    def test_model_directory_from_zoo_2(self, setup):
        self._test_model_directory_from_zoo(
            setup, download_folders_as_tars=True, multiple_sample_outputs=False
        )

    def test_model_directory_from_zoo_3(self, setup):
        with pytest.warns():
            self._test_model_directory_from_zoo(
                setup, download_folders_as_tars=False, multiple_sample_outputs=True
            )

    def test_model_directory_from_zoo_4(self, setup):
        with pytest.warns():
            self._test_model_directory_from_zoo(
                setup, download_folders_as_tars=False, multiple_sample_outputs=False
            )

    @staticmethod
    def _test_model_directory_from_zoo(
        setup, download_folders_as_tars, multiple_sample_outputs
    ):
        request_json, expected_files = setup
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        if download_folders_as_tars:
            request_json = TestModelDirectoryFromZooApi._swap_folders_for_tars(
                request_json
            )
        if multiple_sample_outputs:
            request_json = TestModelDirectoryFromZooApi._clone_sample_outputs(
                request_json
            )
        model_directory = ModelDirectory.from_zoo_api(request_json=request_json)
        assertion_target = (
            set()
            if not multiple_sample_outputs
            else {
                "sample_outputs_deepsparse.tar.gz",
                "sample_outputs_onnxruntime.tar.gz",
            }
        )
        assert model_directory.download(directory_path=temp_dir.name)
        if not download_folders_as_tars:
            expected_files = [
                name.replace(".tar.gz", "")
                if name in ["training.tar.gz", "deployment.tar.gz"]
                else name
                for name in expected_files
            ]
        assert (
            set(os.listdir(temp_dir.name)).difference(expected_files)
            == assertion_target
        )

        shutil.rmtree(temp_dir.name)

    @staticmethod
    def _swap_folders_for_tars(request_json):
        folders_to_swap = ["training", "deployment"]
        folders_to_create = ["logs", "onnx"]

        request_json = [
            file_dict
            for file_dict in request_json
            if file_dict["file_type"] not in folders_to_swap
        ]

        for folder_name in folders_to_create + folders_to_swap:
            # create all archived directories (using `sample-inputs.tar.gz` to simulate)
            sample_inputs_folder_file = fetch_from_request_json(
                request_json, "display_name", "sample_inputs.tar.gz"
            )
            assert len(sample_inputs_folder_file) == 1
            file_dict = copy.copy(sample_inputs_folder_file[0][1])
            file_dict["display_name"] = folder_name + ".tar.gz"
            file_dict["file_type"] = folder_name
            request_json.append(file_dict)
        return request_json

    @staticmethod
    def _clone_sample_outputs(request_json):
        cloned_tars_names = [
            "sample_outputs_deepsparse.tar.gz",
            "sample_outputs_onnxruntime.tar.gz",
        ]
        sample_outputs_folder_file = fetch_from_request_json(
            request_json, "display_name", "sample_outputs.tar.gz"
        )
        assert len(sample_outputs_folder_file) == 1
        idx, file_dict = sample_outputs_folder_file[0]
        for name in cloned_tars_names:
            file_dict_ = copy.copy(file_dict)
            file_dict_["display_name"] = name
            request_json.append(file_dict_)
        del request_json[idx]

        return request_json

    @staticmethod
    def _get_request_json(stub: str):
        model = Zoo.load_model_from_stub(stub)
        request_json = download_model_get_request(args=model)["model"]["files"]
        request_json = restructure_request_json(request_json)
        return request_json
