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


from sparsezoo.git_server_model.models.benchmark import Benchmark
from sparsezoo.git_server_model.models.model_metadata import ModelMetadata
from sparsezoo.git_server_model.validations import BenchmarkValidation, ModelValidation


class Model:
    def __init__(self, model_path: str):
        self.model = ModelMetadata(model_path)
        self.benchmark = Benchmark(model_path)
