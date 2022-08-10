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

from sparsezoo.git_server.utils import get_model_file
from sparsezoo.git_server.validations import BenchmarkValidation


FILE_NAME = "benchmark.yaml"


class Benchmark:
    def __init__(self, path: str, platform: str = "web", branch="main"):
        """
        Get the benchmark metadata

        :param path: path the folder that contains benchmark.yaml
        :param platform: Set to either 'web' or 'local'.
            'web' reads from git server, local reads from
            the local dir
        :param branch: git server branch
        """

        self.benchmarks = get_model_file(
            path=path, file_name=FILE_NAME, platform=platform, branch=branch
        )

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

    @staticmethod
    def validate(path: str, platform: str = "web", branch="main") -> bool:
        """
        Given a benchmark file, validate its metadata

        :param path: path the folder that contains benchmark.yaml
        :param platform: Set to either 'web' or 'local'.
            'web' reads from git server, local reads from
            the local dir
        :param branch: git server branch
        :return: boolean, whether validation (metadata population) passed
        """
        candidate = get_model_file(
            path=path, file_name=FILE_NAME, platform=platform, branch=branch
        )
        try:
            if BenchmarkValidation(**candidate):
                return True
        except Exception as _err:
            print(_err)
            return False
