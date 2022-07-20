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
from pathlib import Path

import numpy
import pytest

from sparsezoo.v2.objects import Directory, File, Model


files_ic = {
    "analysis.yaml",
    "deployment",
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
    "training",
}

files_nlp = copy.copy(files_ic)
files_nlp.remove("recipe_transfer_learn.md")
files_yolo = copy.copy(files_ic)


@pytest.mark.parametrize(
    "stub, args, should_pass",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
            ("recipe", "transfer_learn"),
            True,
        ),
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
            ("recipe", "some_dummy_name"),
            False,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            ("deployment", ""),
            True,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            ("checkpoint", "preqat"),
            True,
        ),
    ],
    scope="function",
)
def test_model_from_stub(stub, args, should_pass):
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
    path = stub + "?" + args[0] + "=" + args[1]
    if should_pass:
        model = Model(path)
        model.download(directory_path=temp_dir.name)
        _assert_correct_files_downloaded(model, args)
    else:
        with pytest.raises(ValueError):
            model = Model(path)
            model.download(directory_path=temp_dir.name)


def _assert_correct_files_downloaded(model, args):
    for file_name, file in model._files_dictionary.items():
        if args[0] == "recipe" and file_name == "recipes":
            assert isinstance(file, File)
        elif args[0] == "deployment" and file_name == "deployment":
            assert isinstance(file, Directory)

        elif args[0] == "checkpoint" and file_name == "training":
            assert isinstance(file, Directory)

        else:
            assert file is None


@pytest.mark.parametrize(
    "stub, clone_sample_outputs, expected_files",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",  # noqa E501
            True,
            files_ic,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            False,
            files_nlp,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",  # noqa E501
            True,
            files_nlp,
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
            True,
            files_yolo,
        ),
    ],
    scope="function",
)
class TestModelDirectory:
    @pytest.fixture()
    def setup(self, stub, clone_sample_outputs, expected_files):
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Model(path=stub)
        model.download(directory_path=temp_dir.name)
        self._add_mock_files(temp_dir.name, clone_sample_outputs=clone_sample_outputs)
        model = Model(path=temp_dir.name)

        yield model, clone_sample_outputs, expected_files, temp_dir

        shutil.rmtree(temp_dir.name)

    def test_folder_structure(self, setup):
        _, _, expected_files, temp_dir = setup
        assert not set(os.listdir(temp_dir.name)).difference(expected_files)

    def test_validate(self, setup):
        model, clone_sample_outputs, _, _ = setup
        assert model.validate(validate_onnxruntime=clone_sample_outputs)
        assert model.validate(
            validate_onnxruntime=clone_sample_outputs, minimal_validation=True
        )

    def test_generate_outputs(self, setup):
        model, clone_sample_outputs, _, _ = setup
        if clone_sample_outputs:
            for engine in ["onnxruntime", "deepsparse"]:
                self._test_generate_outputs_single_engine(engine, model)

    @staticmethod
    def _add_mock_files(directory_path: str, clone_sample_outputs: bool):
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

        # add yaml files
        for file_name in ["analysis.yaml", "benchmarks.yaml", "eval.yaml"]:
            file_dir = os.path.join(directory_path, file_name)
            Path(file_dir).touch()

        # special case for `recipe.yaml` in transformers' tests
        if "tokenizer.json" in os.listdir(os.path.join(directory_path, "training")):
            optional_recipe_yaml = os.path.join(
                directory_path, "training", "recipe.yaml"
            )
            Path(optional_recipe_yaml).touch()

        # add remaining `sample_{...}` files, that may be potentially
        # missing
        mock_sample_file = os.path.join(directory_path, "sample_inputs.tar.gz")
        for file_name in ["sample_originals.tar.gz", "sample_labels.tar.gz"]:
            expected_file_dir = os.path.join(directory_path, file_name)
            if not os.path.isfile(expected_file_dir):
                shutil.copyfile(mock_sample_file, expected_file_dir)

        if clone_sample_outputs:
            sample_outputs_file = os.path.join(directory_path, "sample_outputs.tar.gz")
            for file_name in [
                "sample_outputs_onnxruntime.tar.gz",
                "sample_outputs_deepsparse.tar.gz",
            ]:
                shutil.copyfile(
                    sample_outputs_file, os.path.join(directory_path, file_name)
                )
            os.remove(sample_outputs_file)

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
            # for 'deepsparse' accuracy, we lower the accuracy here (1e-4)
            if engine == "onnxruntime":
                assert numpy.isclose(o1, o2, atol=1e-5).all()
            else:
                assert numpy.isclose(o1, o2, atol=1e-4).all()

        if engine == "onnxruntime":
            assert os.path.isfile(tar_file_expected_path)
