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


from sparsezoo.git_server.models.benchmark import Benchmark
from sparsezoo.git_server.models.modelcard_metadata import ModelCardMetadata


class GitServerModel:
    def __init__(self, model_path: str, platform: str = "web"):
        """
        Class containing git server model info

        :param path: path the folder that contains model files
            (model.md, benchmark.yaml, etc)
        :param platform: Set to either 'web' or 'local'. 'web' reads from
            the git server, local reads from local dir
        :param branch: git server branch
        :return: boolean, whether validation (metadata population) passed
        """

        self.card = ModelCardMetadata(path=model_path, platform=platform)
        self.benchmark = Benchmark(model_path, platform=platform)
