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

from sparsezoo.refactor import Directory, File
from tests.sparsezoo.refactor.test_file import _create_sample_file


def _create_files_directory(files_extensions, temp_dir, dir_name="directory"):
    files = []
    for extension in files_extensions:
        path = tempfile.NamedTemporaryFile(
            delete=False, dir=temp_dir.name, suffix=extension
        )
        name = os.path.basename(path.name)
        files.append(File(name=name, path=path.name))
        _create_sample_file(path.name)

    return dir_name, temp_dir.name, files


@pytest.mark.parametrize(
    "files_extensions", [[".npz", ".onnx", ".md"]], scope="function"
)
class TestDirectory:
    @pytest.fixture()
    def setup(self, files_extensions):

        # base temporary directory
        _temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # second temporary directory (of Directory object)
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

    def test_gzip(self, setup):
        name, path, files = setup
        directory = Directory(name=name, files=files, path=path)
        tar_directory = Directory.gzip(directory)
        tar_name = name + ".tar.gz"
        assert tar_directory.path == os.path.join(os.path.dirname(path), tar_name)
        assert tar_directory.files is None
        assert tar_directory.name == tar_name
        assert os.path.isfile(tar_directory.path)
        assert tar_directory.validate()

    def test_unzip(self, setup):
        name, path, files = setup
        directory = Directory(name=name, files=files, path=path)
        tar_directory = Directory.gzip(directory=directory)
        retrieved_directory = Directory.unzip(
            tar_directory=tar_directory,
            extract_directory=os.path.join(path, "retrieved_directory"),
        )
        assert retrieved_directory.path == os.path.join(path, "retrieved_directory")
        assert retrieved_directory.files is not None
        assert retrieved_directory.name == "retrieved_directory"
        assert os.path.isdir(retrieved_directory.path)
        for file in retrieved_directory.files:
            os.path.isfile(file.path)
            assert file.validate()
        assert retrieved_directory.validate()

    def test_get_file_names(self, setup):
        name, path, files = setup
        directory = Directory(name=name, files=files, path=path)
        assert set([file.name for file in files]) == set(directory.get_file_names())
        tar_directory = Directory.gzip(directory)
        assert set([file.name for file in files]) == set(tar_directory.get_file_names())
