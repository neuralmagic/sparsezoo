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

from typing import Dict, List

from pyndatic import BaseModel
from sparsezoo.analysis.models.benchmark import BenchmarkAnalysis
from sparsezoo.analysis.models.memory_access import MemoryAccessAnalysis
from sparsezoo.analysis.models.operation import OpAnalysis


class NodeAnalysis(BaseModel):
    name: str
    type_: str
    exec_order: int
    graph_order: int
    params: List[str]
    inputs: List[str]
    outputs: List[str]
    ops: OpAnalysis
    mem_access: MemoryAccessAnalysis
    benchmarks: Dict[str, BenchmarkAnalysis]
