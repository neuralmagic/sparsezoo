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

import json
import os
import pathlib
import tempfile
from collections import OrderedDict

import numpy as np
import onnx
import pytest
import yaml
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

from PIL import Image
from src.sparsezoo.objects.model_folder_objects import (
    Directory,
    File,
    FrameworkFiles,
    NumpyDirectory,
    SampleOriginals,
)
from src.sparsezoo.utils.numpy import load_numpy_list, save_numpy


def _create_numpy_file(file_path):
    numpy_input = np.array([1, 2, 3])
    numpy_dict = OrderedDict({"input_0": numpy_input, "input_1": numpy_input})
    file_path, _ = os.path.splitext(
        file_path
    )  # remove extension from the path (required by `save_numpy` function)
    save_numpy(
        array=numpy_dict,
        export_dir=os.path.dirname(file_path),
        name=os.path.basename(file_path),
    )


def _create_onnx_file(file_path):
    node = make_node("MatMul", ["input_0", "input_1"], ["output"], name="test_node")
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, (2, 1)),
        ],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, (1, 1))],
    )
    model = make_model(graph)
    onnx.checker.check_model(model)
    onnx.save_model(model, file_path)


def _create_yaml_file(file_path):
    test_dict = {"test_key": "test_value"}
    with open(file_path, "w") as outfile:
        yaml.dump(test_dict, outfile, default_flow_style=False)


def _create_md_file(file_path):
    test_string = "test_string"
    with open(file_path, "w") as outfile:
        outfile.write(test_string)


def _create_json_file(file_path):
    json_string = "test_string"
    with open(file_path, "w") as outfile:
        json.dump(json_string, outfile)


def _create_image_file(file_path):
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).convert("RGB").save(
        file_path
    )


def _create_csv_file(file_path):
    _create_md_file(file_path)


def _create_sample_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension == ".npz":
        _create_numpy_file(file_path)
    elif extension == ".onnx":
        _create_onnx_file(file_path)
    elif extension == ".yaml":
        _create_yaml_file(file_path)
    elif extension == ".md":
        _create_md_file(file_path)
    elif extension == ".json":
        _create_json_file(file_path)
    elif extension in [".jpg", ".png", ".jpeg"]:
        _create_image_file(file_path)
    elif extension == ".csv":
        _create_csv_file(file_path)


def _insert_directory_into_path(path: str, directory_str: str, index: int = 4):
    # e.g. if   path = '/this/is/path/to/a/file'
    #           directory = 'test'
    #           index = 2
    #           result '/this/test/is/path/to/a/file
    path_parts = list(pathlib.PurePath(path).parts)
    path_parts.insert(index, directory_str)
    return str(pathlib.PurePath("").joinpath(*path_parts))


@pytest.mark.parametrize(
    "extension, is_loadable",
    [
        (".npz", True),
        (".onnx", True),
        (".yaml", True),
        (".md", True),
        (".json", True),
        (".csv", True),
        (".jpg", True),
        (".png", True),
        (".jpeg", True),
        ("", False),
        (".bin", False),
    ],
    scope="function",
)
class TestFile:
    @pytest.fixture()
    def setup(self, extension, is_loadable):
        # setup
        _, path = tempfile.mkstemp(suffix=extension)
        _create_sample_file(path)

        yield path, is_loadable

        # teardown
        os.remove(path)

    def test_validate(self, setup):
        path, is_loadable = setup
        file = File(name="sample_file", path=path)
        assert is_loadable == file.validate()


@pytest.mark.parametrize(
    "files_extensions", [([".npz", ".onnx", ".yaml"])], scope="function"
)
class TestDirectory:
    @pytest.fixture()
    def setup(self, files_extensions):
        files = []
        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)
        for extension in files_extensions:
            path = tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir.name, suffix=extension
            )
            _create_sample_file(path.name)
            files.append(File(name="sample_file", path=path.name))

        yield files, temp_dir

        _temp_dir.cleanup()

    def test_directory(self, setup):
        files, temp_dir = setup
        directory = Directory(name="sample_directory", files=files)
        assert directory.path == temp_dir.name

    def test_gzip(self, setup):
        files, temp_dir = setup
        directory = Directory(name="sample_directory", files=files)
        tar_file_path = directory.gzip()
        assert tar_file_path == os.path.join(
            os.path.dirname(temp_dir.name), f"{directory.name}.tar.gz"
        )
        assert os.path.isfile(tar_file_path)

    def test_unzip(self, setup):
        current_working_directory = os.getcwd()
        files, temp_dir = setup
        directory = Directory(name="sample_directory", files=files)
        with pytest.raises(Exception):
            directory.unzip()
        tar_file_path = directory.gzip()
        directory = Directory(name="sample_tar_directory", path=tar_file_path)
        os.chdir(os.path.dirname(tar_file_path))
        files_before = [
            path.name
            for path in pathlib.Path(os.path.dirname(temp_dir.name)).rglob("*")
            if "." in path.name
        ]
        assert directory.unzip()
        files_after = [
            path.name
            for path in pathlib.Path(os.path.dirname(temp_dir.name)).rglob("*")
            if "." in path.name
        ]
        os.chdir(current_working_directory)
        assert len(files) == len(files_after) - len(files_before)


@pytest.mark.parametrize(
    "files_extensions, directories, files_inside_a_directory",
    [
        (
            [".npz", ".onnx", ".yaml"],
            ["checkpoint_12/", "checkpoint_3/", "logs/"],
            False,
        ),
        (
            [".npz", ".onnx", ".yaml"],
            ["checkpoint_12/", "checkpoint_3/", "logs/"],
            True,
        ),
    ],
    scope="function",
)
class TestFrameworkFiles:
    @pytest.fixture()
    def setup(self, files_extensions, directories, files_inside_a_directory):
        files = []
        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)
        for idx, (extension, directory) in enumerate(
            zip(files_extensions, directories)
        ):
            path = tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir.name, suffix=extension
            )
            # we need to insert `directory` string into the path of the File
            extended_path = _insert_directory_into_path(
                path=path.name, directory_str=directory
            )
            # create directory and the file in it
            os.mkdir(os.path.dirname(extended_path))

            _create_sample_file(extended_path)
            files.append(File(name=f"sample_file_{idx}", path=extended_path))

        if files_inside_a_directory:
            # wrap a FrameworkFiles object around the files
            files = [
                FrameworkFiles(name="sample_directory", path=temp_dir.name, files=files)
            ]

        yield files, temp_dir, files_inside_a_directory

        _temp_dir.cleanup()

    def test_validate(self, setup):
        files, temp_dir, _ = setup
        directory = FrameworkFiles(name="sample_framework_files", files=files)
        assert directory.validate()

    def test_get_file(self, setup):
        file_name1 = "sample_file_2"  # should be found within the FrameworkFiles
        file_name2 = (
            "non_existent_file"  # should not be found within the FrameworkFiles
        )
        directory_name = "sample_directory"

        files, temp_dir, files_inside_a_directory = setup
        framework_files = FrameworkFiles(name="sample_framework_files", files=files)

        file = framework_files.get_file(file_name=file_name1)
        assert file.name == file_name1
        file = framework_files.get_file(file_name=file_name2)
        assert file is None
        if files_inside_a_directory:
            directory = framework_files.get_file(file_name=directory_name)
            assert directory.name == directory_name


@pytest.mark.parametrize(
    "files_extensions", [([".npz", ".onnx", ".yaml"])], scope="function"
)
class TestSampleOriginals:
    @pytest.fixture()
    def setup(self, files_extensions):
        files = []
        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)
        for extension in files_extensions:
            path = tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir.name, suffix=extension
            )
            _create_sample_file(path.name)
            files.append(File(name="sample_file", path=path.name))

        yield files, temp_dir

        _temp_dir.cleanup()

    def test_validate(self, setup):
        files, temp_dir = setup
        sample_originals = SampleOriginals(name="sample_sample_originals", files=files)
        assert sample_originals.validate()

    def test_iter(self, setup):
        files, temp_dir = setup
        sample_originals = SampleOriginals(name="sample_sample_originals", files=files)
        for file_original, file in zip(files, sample_originals):
            assert file_original == file


@pytest.mark.parametrize(
    "files_extensions", [([".npz", ".npz", ".npz"])], scope="function"
)
class TestNumpyDirectory:
    @pytest.fixture()
    def setup(self, files_extensions):
        files = []
        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)
        for extension in files_extensions:
            path = tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir.name, suffix=extension
            )
            _create_sample_file(path.name)
            files.append(File(name="sample_file", path=path.name))

        onnx_model_path = tempfile.NamedTemporaryFile(
            delete=False, dir=temp_dir.name, suffix=".onnx"
        )
        _create_onnx_file(onnx_model_path.name)
        onnx_model = onnx.load(onnx_model_path.name)

        yield files, temp_dir, onnx_model

        _temp_dir.cleanup()

    def test_validate(self, setup):
        files, temp_dir, onnx_model = setup
        numpy_directory = NumpyDirectory(name="sample_numpy_directory", files=files)
        assert numpy_directory.validate(model=onnx_model)

    def test_iter(self, setup):
        files, temp_dir, _ = setup
        numpy_directory = NumpyDirectory(name="sample_numpy_directory", files=files)
        for file_original, numpy_iter in zip(files, numpy_directory):
            numpy_dictionary = load_numpy_list(file_original.path)[0]
            for (key_1, value_1), (key_2, value_2) in zip(
                numpy_dictionary.items(), numpy_iter.items()
            ):
                assert key_1 == key_2  # compare key names
                np.testing.assert_array_equal(value_1, value_2)  # compare numpy arrays
