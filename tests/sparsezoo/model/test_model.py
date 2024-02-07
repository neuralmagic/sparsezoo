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
import tarfile
import tempfile
from pathlib import Path

import numpy
import pytest

from sparsezoo import Model


files_ic = {
    "training",
    "deployment.tar.gz",
    "deployment",
    "logs",
    "onnx",
    "model.onnx",
    "model.onnx.tar.gz",
    "sample-inputs.tar.gz",
    "sample-originals.tar.gz",
    "sample-labels.tar.gz",
    "sample-outputs.tar.gz",
    "sample-inputs",
    "sample-originals",
    "sample-labels",
    "sample-outputs",
    "benchmarks.yaml",
    "benchmark.yaml",
    "eval.yaml",
    "analysis.yaml",
    "model.md",
    "metrics.yaml",
}

files_nlp = copy.copy(files_ic)
files_yolo = copy.copy(files_ic)


@pytest.mark.parametrize(
    "stub, args, should_pass",
    [
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/"
            "pruned-moderate",
            ("recipe", "transfer_learn"),
            True,
        ),
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/"
            "pruned-moderate",
            ("checkpoint", "some_dummy_name"),
            False,
        ),
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/"
            "pruned-moderate",
            ("deployment", "default"),
            True,
        ),
        (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/"
            "pruned-moderate",
            ("checkpoint", "preqat"),
            True,
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/"
            "12layer_pruned80_quant-none-vnni",
            ("checkpoint", "postqat"),
            True,
        ),
        (
            "biobert-base_cased-jnlpba_pubmed-pruned80.4block_quantized",
            ("deployment", "default"),
            True,
        ),
        (
            "resnet_v1-50-imagenet-pruned95",
            ("checkpoint", "preqat"),
            True,
        ),
    ],
    scope="function",
)
class TestSetupModel:
    @pytest.fixture()
    def setup(self, stub, args, should_pass, tmpdir):
        yield stub, args, should_pass, tmpdir

    def test_model_from_stub(self, setup):
        stub, args, should_pass, tmpdir = setup
        path = stub + "?" + args[0] + "=" + args[1]
        if should_pass:
            model = Model(path, tmpdir)
            self._assert_correct_files_downloaded(model, args)
            self._assert_validation_results_exist(model)
            assert model.compressed_size
        else:
            with pytest.raises(ValueError):
                Model(path)

    @staticmethod
    def _assert_correct_files_downloaded(model, args):
        if args[0] == "checkpoint":
            assert len(model.training.available) == 1
        elif args[0] == "deployment":
            assert len(model.training.available) == 1

    @staticmethod
    def _assert_validation_results_exist(model):
        assert model.validation_results is not None
        assert isinstance(model.validation_results, dict)
        assert len(model.validation_results.keys()) >= 1
        assert any(value for value in model.validation_results.values())


@pytest.mark.parametrize(
    "stub, clone_sample_outputs, expected_files",
    [
        (
            (
                "zoo:"
                "cv/classification/mobilenet_v1-1.0/"
                "pytorch/sparseml/imagenet/pruned-moderate"
            ),
            True,
            files_ic.union({"recipe.md", "recipe_transfer_learn.md"}),
        ),
        (
            (
                "zoo:"
                "nlp/question_answering/distilbert-none/"
                "pytorch/huggingface/squad/pruned80_quant-none-vnni"
            ),
            False,
            files_nlp.union({"recipe.md"}),
        ),
        (
            (
                "zoo:"
                "cv/detection/yolov5-s/"
                "pytorch/ultralytics/coco/pruned_quant-aggressive_94"
            ),
            True,
            files_yolo.union({"recipe.md", "recipe_transfer_learn.md"}),
        ),
        (
            "yolov5-x-coco-pruned70.4block_quantized",
            False,
            files_yolo.union({"recipe.md", "recipe_transfer_learn.md"}),
        ),
        (
            "yolov5-n6-voc_coco-pruned55",
            False,
            files_yolo.union({"recipe.md"}),
        ),
        (
            "resnet_v1-50-imagenet-channel30_pruned90_quantized",
            False,
            files_yolo.union({"recipe.md", "recipe_transfer_classification.md"}),
        ),
    ],
    scope="function",
)
class TestModel:
    @pytest.fixture()
    def setup(self, stub, clone_sample_outputs, expected_files):
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        model = Model(stub, temp_dir.name)
        model.download()
        self._add_mock_files(temp_dir.name, clone_sample_outputs=clone_sample_outputs)
        model = Model(temp_dir.name)

        yield model, clone_sample_outputs, expected_files, temp_dir

        shutil.rmtree(temp_dir.name)

    def test_folder_structure(self, setup):
        _, clone_sample_outputs, expected_files, temp_dir = setup
        if clone_sample_outputs:
            for file_name in [
                "sample-outputs_onnxruntime",
                "sample-outputs_deepsparse",
            ]:
                expected_files.update({file_name, file_name + ".tar.gz"})
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
            self._test_generate_outputs_single_engine("onnxruntime", model)

    @staticmethod
    def _add_mock_files(directory_path: str, clone_sample_outputs: bool):
        # add some mock files, to complete the full set of
        # possible expected files in the `Model`
        # class object

        # add onnx directory
        onnx_folder_dir = os.path.join(directory_path, "onnx")
        os.makedirs(onnx_folder_dir)
        for opset in range(1, 3):
            shutil.copyfile(
                os.path.join(directory_path, "deployment", "model.onnx"),
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

        # add remaining `sample-{...}` files, that may be potentially
        # missing
        mock_sample_file = os.path.join(directory_path, "sample-inputs.tar.gz")
        for file_name in ["sample-originals.tar.gz", "sample-labels.tar.gz"]:
            expected_file_dir = os.path.join(directory_path, file_name)
            if not os.path.isfile(expected_file_dir):
                shutil.copyfile(mock_sample_file, expected_file_dir)

        if clone_sample_outputs:
            sample_outputs_file = os.path.join(directory_path, "sample-outputs.tar.gz")
            for file_name in [
                "sample-outputs_onnxruntime.tar.gz",
                "sample-outputs_deepsparse.tar.gz",
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
                directory_path, f"sample-outputs_{engine}.tar.gz"
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


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
        "imagenet/pruned-moderate",
    ],
)
def test_model_gz_extraction_from_stub(stub: str):
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")

    model = Model(stub, temp_dir.name)
    _extraction_test_helper(model)
    shutil.rmtree(temp_dir.name)


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
        "imagenet/pruned-moderate",
    ],
)
def test_model_gz_extraction_from_local_files(stub: str):
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
    model = Model(stub, temp_dir.name)
    model.download()

    source = temp_dir.name
    model_from_local_files = Model(source)
    _extraction_test_helper(model_from_local_files)
    shutil.rmtree(temp_dir.name)


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/"
        "imagenet/pruned-moderate",
    ],
)
def _extraction_test_helper(model: Model):
    # download and extract model.onnx.tar.gz
    #  path should point to extracted model.onnx file
    path = Path(model.onnx_model.path)

    # assert path points to model.onnx file
    assert path.exists(), f"{path} does not exist"
    assert path.is_file(), f"{path} is not a file"

    # assert model.onnx.tar.gz exists
    model_gz_path = path.with_name("model.onnx.tar.gz")
    assert model_gz_path.exists(), f"{model_gz_path} does not exist"

    # assert all members of  model.onnx.tar.gz have been extracted
    for zipped_filename in tarfile.open(model_gz_path).getnames():
        unzipped_file_path = path.with_name(zipped_filename)
        assert (
            unzipped_file_path.exists()
        ), f"{unzipped_file_path} does not exist, was it extracted?"
