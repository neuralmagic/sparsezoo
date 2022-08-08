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

from pyexpat import model
from sparsezoo.git_server_model.utils import local_load, web_load
from sparsezoo.git_server_model.validations import ModelCardValidation


class GitServerModelCardMetadata:
    def __init__(self, path: str, platform: str = "web"):
        if platform == "local":
            self.model_card = local_load(path, filename="model.md")
        else:
            self.model_card = web_load(path, filename="model.md")

        """NEED VERSION 1.0 of model.md to test 
       
        Then
            1. ModelCardValidation, uncomment fields
            2. Uncomment below
            3. Copy model.md and make tests
        """

        self.metadata = ModelCardValidation(**self.model_card).dict()

        self.card_version = self.metadata["card_version"]
        # self.base = self.metadata["base"]
        self.parent = self.metadata["parent"]
        self.domain = self.metadata["domain"]
        self.task = self.metadata["task"]
        self.architecture = self.metadata["architecture"]
        self.task = self.metadata["task"]
        self.framework = self.metadata["framework"]
        self.repo = self.metadata["repo"]
        self.source_dataset = self.metadata["source_dataset"]
        # self.train_dataset = self.metadata["train_dataset"]
        # self.optimizations = self.metadata["optimizations"]
        self.display_name = self.metadata["display_name"]
        self.tags = self.metadata["tags"]
        # self.commands = self.metadata["commands"]

    # Maybe can make into more general if have time, instead of validate method for each class
    @staticmethod
    def validate(path: str):
        """
        Given a model card in development, validate its metadata
        """
        candidate = local_load(path, filename="model.md")
        try:
            if ModelCardValidation(**candidate):
                return True
        except Exception as _err:
            print(_err)
            return False

    # Deep validation, not required for the currrent task
    """
    Make DB connection, check if entries exist
    Check if AWS and gitserver entries are updated to the latest
   
    """
