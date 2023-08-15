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

from sparsezoo.objects import Directory, File
from tests.sparsezoo.objects.test_file import _create_sample_file


def _create_files_directory(files_extensions, temp_dir):
    files = []
    for extension in files_extensions:
        path = tempfile.NamedTemporaryFile(
            delete=False, dir=temp_dir.name, suffix=extension
        )
        name = os.path.basename(path.name)
        files.append(File(name=name, path=path.name))
        _create_sample_file(path.name)

    return os.path.basename(temp_dir.name), temp_dir.name, files


@pytest.mark.parametrize(
    "files_extensions", [[".npz", ".onnx", ".md"]], scope="function"
)
class TestDirectory:
    @pytest.fixture()
    def setup(self, files_extensions):

        # base temporary directory
        __temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
        _temp_dir = tempfile.TemporaryDirectory(dir=__temp_dir.name)
        # third temporary directory (of Directory object)
        # (for some tests we nest directories up to the third level)
        temp_dir = tempfile.TemporaryDirectory(dir=_temp_dir.name)

        name, path, list_files = _create_files_directory(files_extensions, temp_dir)

        yield name, path, list_files

        _temp_dir.cleanup()

    def test_directory(self, setup):
        (
            name,
            path,
            files,
        ) = setup
        directory = Directory(name=name, files=files, path=path)
        assert directory.path == path
        assert directory.files == files
        assert directory.name == name
        assert directory.validate()

    def test_nested_directory(self, setup):
        (
            name,
            path,
            files,
        ) = setup
        directory = Directory(name=name, files=files, path=path)

        for _ in range(2):
            directory = self._create_nested_directory(directory)
        assert directory.path == os.path.dirname(os.path.dirname(path))
        assert directory.name == os.path.basename(
            os.path.dirname(os.path.dirname(path))
        )
        assert directory.validate()

    def test_nested_directory_from_files(self, setup):
        # tests whether File class objects, which represent directories,
        # get automatically converted into directories
        (
            name,
            path,
            files,
        ) = setup
        file = File(name=name, path=path)

        for _ in range(2):
            file = self._create_nested_directory(file)
        directory = Directory.from_file(file)
        assert directory.path == os.path.dirname(os.path.dirname(path))
        assert directory.name == os.path.basename(
            os.path.dirname(os.path.dirname(path))
        )
        assert directory.validate()

    def test_zipping_on_creation(self, setup):
        (
            name,
            path,
            files,
        ) = setup
        directory = Directory(name=name, files=files, path=path)
        directory.gzip()
        new_directory = Directory(name=directory.name, path=directory.path, force=True)
        assert os.path.isdir(new_directory.path)
        assert new_directory.path == directory.path.replace(".tar.gz", "")
        assert new_directory.files
        assert new_directory.name == directory.name.replace(".tar.gz", "")
        assert not new_directory.is_archive
        assert new_directory.validate()

    def test_gzip(self, setup):
        name, path, files = setup
        directory = Directory(name=name, files=files, path=path)
        name = directory.name
        directory.gzip()
        tar_name = name + ".tar.gz"
        assert os.path.isfile(directory.path)

        assert directory.path == os.path.join(os.path.dirname(path), tar_name)
        assert not directory.files
        assert directory.name == tar_name
        assert directory.is_archive
        assert directory.validate()

    def test_unzip(self, setup):
        name, path, files = setup
        directory = Directory(name=name, files=files, path=path)
        directory.gzip()
        shutil.rmtree(path)
        directory.unzip()
        assert os.path.isdir(directory.path)

        assert directory.path == os.path.join(os.path.dirname(path), name)
        assert all([x.name == y.name for x, y in zip(directory.files, files)])
        assert directory.name == name
        assert not directory.is_archive
        assert directory.validate()

    def test_get_file_names(self, setup):
        name, path, files = setup
        directory = Directory(name=name, files=files, path=path)
        assert set([file.name for file in files]) == set(directory.get_file_names())
        directory.gzip()
        assert set([file.name for file in files]) == set(directory.get_file_names())

    def test_get_file(self, setup):
        name, path, files = setup
        directory = Directory(name=name, files=files, path=path)

        file_name_exists = directory.files[0].name
        file_name_not_exists = "dummy.onnx"

        assert directory.get_file(file_name=file_name_exists)
        assert directory.get_file(file_name=file_name_not_exists) is None

        for _ in range(2):
            directory = self._create_nested_directory(directory)

        assert directory.get_file(file_name=file_name_exists)
        assert directory.get_file(file_name=file_name_not_exists) is None

    def _create_nested_directory(
        self, file_or_directory: File, nesting_width: int = 3
    ) -> File:
        new_files = [file_or_directory] * nesting_width
        new_path = os.path.dirname(file_or_directory.path)
        new_name = os.path.basename(new_path)
        if isinstance(file_or_directory, Directory):
            return Directory(name=new_name, files=new_files, path=new_path)
        else:
            return File(name=new_name, path=new_path)
