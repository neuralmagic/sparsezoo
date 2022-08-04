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

import re
from typing import Dict

import requests
import yaml

from pyexpat import model
from sparsezoo.git_server_model.utils import extract_git_server_metadata


class ModelMetadata:
    def __init__(self, path):
        metadata = self.local_load(path)

    @staticmethod
    def local_load(model_card_path: str):
        """Given a model.md file, return a dict with its contents as metadata"""

        with open(model_card_path, "r") as yaml_file:
            model_card_metadata = next(yaml.safe_load_all(yaml_file.read()))
        return model_card_metadata

    @staticmethod
    def web_load(git_server_url: str, filename: str):
        """
        Given a gitserver url (url of the gitserver model, or git clone ssh/http),
        and the file name, get the contents of the file using http request
        """
        git_server_metadata = extract_git_server_metadata(git_server_url)

        raw_file_content = requests.get()
        pass

        # https://git.neuralmagic.com/neuralmagic/cary/-/raw/main/model.md


# with open("example.yaml", "r") as stream:
#     try:
#         print(yaml.safe_load(stream))
#     except yaml.YAMLError as exc:
#         print(exc)
