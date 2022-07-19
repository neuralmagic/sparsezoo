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

import tempfile

import numpy as np
import onnx
import pytest

from sparsezoo.utils.numpy import load_numpy_list
from sparsezoo.v2.objects import NumpyDirectory
from tests.sparsezoo.v2.objects.test_directory import _create_files_directory
from tests.sparsezoo.v2.objects.test_file import _create_onnx_file


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
