# from sparsezoo.git_server_model import Model

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

from sparsezoo.git_server_model.utils import local_load, web_load
from sparsezoo.git_server_model.validations import BenchmarkValidation


class Benchmark:
    def __init__(self, path: str, platform: str = "web"):
        if platform == "local":
            self.benchmarks = local_load(folder_path=path, filename="benchmark.yaml")
        else:
            self.benchmarks = web_load(git_server_url=path, filename="benchmark.yaml")

        self.metadata = BenchmarkValidation(**self.benchmarks).dict()
        self.deepsparse_version = self.metadata["benchmarks"]["repo"][
            "deepsparse_version"
        ]
        self.git_ssh_url = self.metadata["benchmarks"]["repo"]["git_ssh_url"]
        self.model_commit_sha = self.metadata["benchmarks"]["repo"]["model_commit_sha"]
        self.sample_input_commit_sha = self.metadata["benchmarks"]["repo"][
            "sample_input_commit_sha"
        ]
        self.results = self.metadata["benchmarks"]["results"]

    pass
