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

from pydantic import BaseModel, Field, validator

from sparsezoo.analyze_v2.model_validator.memory_access_analysis_model import (
    MemoryAccessAnalysisModel,
)
from sparsezoo.analyze_v2.model_validator.operation_analysis_model import (
    OperationAnalysisModel,
)
from sparsezoo.analyze_v2.model_validator.parameter_analysis_model import (
    ParameterAnalysisModel,
)


class NodeAnalysisModel(BaseModel):
    name: str = Field(..., description="Node name")
    op_type: str = Field(..., description="Node op type")
    graph_order: int
    input: List[str]
    output: List[str]
    ops: OperationAnalysisModel
    params: ParameterAnalysisModel
    mem_access: MemoryAccessAnalysisModel

    @validator("input", "output", pre=True)
    def validate_types(cls, value):
        return [val for val in value]
