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

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from sparsezoo.analyze_v2.schemas.memory_access_analysis import (
    MemoryAccessAnalysisSchema,
)
from sparsezoo.analyze_v2.schemas.operation_analysis import OperationAnalysisSchema
from sparsezoo.analyze_v2.schemas.parameter_analysis import ParameterAnalysisSchema


class NodeAnalysisSchema(BaseModel):
    name: str = Field(..., description="Node name")
    op_type: str = Field(..., description="Node op type")
    graph_order: int
    input: List[str]
    output: List[str]
    ops: Optional[OperationAnalysisSchema] = None
    params: ParameterAnalysisSchema
    mem_access: MemoryAccessAnalysisSchema

    @field_validator("input", "output", mode="before")
    @classmethod
    def validate_types(cls, value):
        return [val for val in value]
