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
import pathlib
import tempfile

import numpy as np
import onnx
import pytest

from sparsezoo.refactor import File, FrameworkFiles, NumpyDirectory, SampleOriginals
from sparsezoo.utils.numpy import load_numpy_list
from tests.sparsezoo.refactor.test_directory import _create_files_directory
from tests.sparsezoo.refactor.test_file import _create_onnx_file, _create_sample_file


def _insert_directory_into_path(path: str, directory_str: str, index: int = 4):
    # e.g. if   path = '/this/is/path/to/a/file'
    #           directory = 'test'
    #           index = 2
    #           result '/this/test/is/path/to/a/file
    path_parts = list(pathlib.PurePath(path).parts)
    path_parts.insert(index, directory_str)
    return str(pathlib.PurePath("").joinpath(*path_parts))


@pytest.mark.parametrize(
    "files_extensions, dir_extensions",
    [
        (
            [".npz", ".onnx", ".md"],
            ["checkpoint_12/", "checkpoint_3/", "logs/"],
        ),
    ],
    scope="function",
)
class TestFrameworkFiles:
    @pytest.fixture()
    def setup(self, files_extensions, dir_extensions):

        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)

        name, path, list_files = self._create_files_directory(
            files_extensions, dir_extensions, temp_dir
        )

        # wrap a FrameworkFiles object around the files
        # files = [
        #    FrameworkFiles(name="sample_directory", path=temp_dir.name, files=files)
        # ]

        yield name, path, list_files

        _temp_dir.cleanup()

    @staticmethod
    def _create_files_directory(file_extensions, dir_extensions, temp_dir):
        files = []
        for idx, (extension, directory) in enumerate(
            zip(file_extensions, dir_extensions)
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

            # the directory and file are created, create related File object
            name = os.path.basename(path.name)
            files.append(File(name=name, path=extended_path))

        return "framework_files", temp_dir.name, files

    def test_validate(self, setup):
        name, path, files = setup
        framework_files = FrameworkFiles(name=name, files=files, path=path)
        assert framework_files.path == path
        assert framework_files.files == files
        assert framework_files.name == name
        assert framework_files.validate()

        name = "nested_framework_files"
        path = os.path.join(path, name)
        files = [framework_files]

        nested_framework_files = FrameworkFiles(name=name, files=files, path=path)
        assert nested_framework_files.path == path
        assert nested_framework_files.files == files
        assert nested_framework_files.name == name
        assert nested_framework_files.validate()

    def test_get_file(self, setup):
        name, path, files = setup

        framework_files = FrameworkFiles(name=name, files=files, path=path)

        file_name_exists = framework_files.files[0].name
        file_name_not_exists = "dummy.onnx"

        assert framework_files.get_file(file_name=file_name_exists)
        assert framework_files.get_file(file_name=file_name_not_exists) is None

        name = "nested_framework_files"
        path = os.path.join(path, name)
        files = [framework_files]

        nested_framework_files = FrameworkFiles(name=name, files=files, path=path)
        file_name_exists_ = framework_files.name
        assert nested_framework_files.get_file(file_name=file_name_exists)
        assert nested_framework_files.get_file(file_name=file_name_exists_)
        assert nested_framework_files.get_file(file_name=file_name_not_exists) is None


@pytest.mark.parametrize(
    "files_extensions", [([".npz", ".onnx", ".md"])], scope="function"
)
class TestSampleOriginals:
    @pytest.fixture()
    def setup(self, files_extensions):

        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)

        name, path, list_files = _create_files_directory(files_extensions, temp_dir)

        yield name, path, list_files

        _temp_dir.cleanup()

    def test_sample_originals(self, setup):
        (
            name,
            path,
            files,
        ) = setup
        sample_originals = SampleOriginals(name=name, files=files, path=path)
        assert sample_originals.path == path
        assert sample_originals.files == files
        assert sample_originals.name == name
        assert sample_originals.validate()

    def test_iterate(self, setup):
        (
            name,
            path,
            files,
        ) = setup
        sample_originals = SampleOriginals(name=name, files=files, path=path)

        for sample_original, file in zip(files, sample_originals):
            assert sample_original == file


@pytest.mark.parametrize(
    "files_extensions", [([".npz", ".npz", ".npz"])], scope="function"
)
class TestNumpyDirectory:
    @pytest.fixture()
    def setup(self, files_extensions):
        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)

        name, path, list_files = _create_files_directory(files_extensions, temp_dir)

        onnx_model = self._create_onnx_model(temp_dir)

        yield name, path, list_files, onnx_model

        _temp_dir.cleanup()

    def test_validate(self, setup):
        name, path, files, onnx_model = setup
        name = "sample-inputs"
        numpy_directory = NumpyDirectory(name=name, files=files, path=path)
        assert numpy_directory.path == path
        assert numpy_directory.files == files
        assert numpy_directory.name == name
        assert numpy_directory.validate(onnx_model)

    def test_iter(self, setup):
        name, path, files, onnx_model = setup
        numpy_directory = NumpyDirectory(name=name, files=files, path=path)
        for file, numpy_dict in zip(files, numpy_directory):
            numpy_dict_tgt = load_numpy_list(load_numpy_list(file.path))[0]
            for (key_1, value_1), (key_2, value_2) in zip(
                numpy_dict_tgt.items(), numpy_dict.items()
            ):
                assert key_1 == key_2  # compare key names
                np.testing.assert_array_equal(value_1, value_2)  # compare numpy arrays

    @staticmethod
    def _create_onnx_model(temp_dir):
        onnx_model_path = tempfile.NamedTemporaryFile(
            delete=False, dir=temp_dir.name, suffix=".onnx"
        )
        _create_onnx_file(onnx_model_path.name)
        onnx_model = onnx.load(onnx_model_path.name)
        return onnx_model
