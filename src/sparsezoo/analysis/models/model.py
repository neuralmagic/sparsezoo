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

from typing import Dict

from pydantic import BaseModel

from sparsezoo.analysis.models.activation import ActivationsAnalysis
from sparsezoo.analysis.models.benchmark import BenchmarkAnalysis
from sparsezoo.analysis.models.node import NodeAnalysis
from sparsezoo.analysis.models.parameter import ParamAnalysis
from sparsezoo.analysis.models.summary import SummaryAnalysis


class ModelAnalysis(BaseModel):
    name: str
    type_: str
    summaries: Dict[
        str, SummaryAnalysis
    ]  # params, ops, mem_access -> frequently accessed for the SparseZoo web
    params: Dict[str, ParamAnalysis]  # = None if detailed=False
    activations: Dict[str, ActivationsAnalysis]  # None if detailed=False
    nodes: Dict[str, NodeAnalysis]  # = None if detailed=False
    benchmarks: Dict[str, BenchmarkAnalysis]  # future work
