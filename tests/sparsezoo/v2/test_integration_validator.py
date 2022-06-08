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

from sparsezoo import Zoo
from sparsezoo.v2.model_directory import ModelDirectory


class TestIntegrationValidator:
    @pytest.fixture()
    def setup(self):  # domain, sub_domain, model_index):
        # setup
        # temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # model = Zoo.search_models(
        #     domain=domain, sub_domain=sub_domain, override_folder_name=temp_dir.name
        # )[model_index]
        # directory_path = self._get_local_directory(model)
        directory_path = "/home/damian/folder"

        yield directory_path, None  # temp_dir

        # temp_dir.cleanup()

    def test_model_directory_from_directory(self, setup):
        directory_path, temp_dir = setup
        model_directory = ModelDirectory.from_directory(directory_path=directory_path)
        assert model_directory.validate()
