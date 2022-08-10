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

from sparsezoo.git_server.utils import get_model_file
from sparsezoo.git_server.validations import ModelCardValidation


FILE_NAME = "model.md"


class ModelCardMetadata:
    def __init__(self, path: str, platform: str = "web", branch="main"):
        """
        Model metadata population from the model_card

        :param path: path the folder that contains model.md
        :param platform: Set to either 'web' or 'local'. 'web' reads from git server,
            local reads from local dir
        :param branch: git server branch
        :return: boolean, whether validation (metadata population) passed
        """

        self.model_card = get_model_file(
            path=path, file_name=FILE_NAME, platform=platform, branch=branch
        )

        self.metadata = ModelCardValidation(**self.model_card).dict()

        self.card_version = self.metadata["card_version"]
        self.base = self.metadata["base"]
        self.parent = self.metadata["parent"]
        self.domain = self.metadata["domain"]
        self.task = self.metadata["task"]
        self.architecture = self.metadata["architecture"]
        self.task = self.metadata["task"]
        self.framework = self.metadata["framework"]
        self.repo = self.metadata["repo"]
        self.source_dataset = self.metadata["source_dataset"]
        self.train_dataset = self.metadata["train_dataset"]
        self.optimizations = self.metadata["optimizations"]
        self.display_name = self.metadata["display_name"]
        self.tags = self.metadata["tags"]
        self.commands = self.metadata["commands"]

    @staticmethod
    def validate(path: str, platform: str = "web", branch="main") -> bool:
        """
        Given a model card in development, validate its metadata
        """

        candidate = get_model_file(
            path=path, file_name=FILE_NAME, platform=platform, branch=branch
        )
        try:
            if ModelCardValidation(**candidate):
                return True
        except Exception as _err:
            print(_err)
            return False
