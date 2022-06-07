import pytest
import os
import tempfile
from sparsezoo import Zoo
from sparsezoo.v2.model_directory import ModelDirectory

class TestIntegrationValidator:
    @pytest.fixture()
    def setup(self):# domain, sub_domain, model_index):
        # setup
        # temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        # model = Zoo.search_models(
        #     domain=domain, sub_domain=sub_domain, override_folder_name=temp_dir.name
        # )[model_index]
        #directory_path = self._get_local_directory(model)
        directory_path = "/home/damian/folder"

        yield directory_path, None #temp_dir

        #temp_dir.cleanup()

    def test_model_directory_from_directory(self, setup):
        directory_path, temp_dir = setup
        model_directory = ModelDirectory.from_directory(directory_path=directory_path)
        assert model_directory.validate()



