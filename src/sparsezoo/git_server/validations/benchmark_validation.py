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

from typing import List

from pydantic import BaseModel, validator


class BenchmarkStatistics(BaseModel):
    items_per_sec: float
    mean: float
    median: float
    scenario: str
    seconds_ran: float
    std: float


class BenchmarkRepo(BaseModel):
    deepsparse_version: str
    git_ssh_url: str
    model_commit_sha: str
    sample_input_commit_sha: str


class BenchmarkResults(BaseModel):
    batch_size: int
    benchmark_result: BenchmarkStatistics
    engine: str
    input_shapes: List[List[int]]
    instance_type: str
    num_cores: int
    num_streams: int
    onnx_file_path: str
    version: str

    @validator("batch_size", "num_cores", "num_streams", always=True)
    def validate_int_type(cls, value):
        if isinstance(value, int):
            return value
        return int(value)


class BenchmarkFields(BaseModel):
    repo: BenchmarkRepo
    results: List[BenchmarkResults]


class BenchmarkValidation(BaseModel):
    benchmarks: BenchmarkFields
