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

from typing import Optional

from pydantic import BaseModel, Field, validator

from sparsezoo.analyze_v2.model_validator.utils import type_validator


class SparsityAnalysisModel(BaseModel):
    grouping: str = Field(..., description="The combining group name")
    counts: int = Field(..., description="Total number of parameters in the node")
    counts_sparse: int = Field(
        ..., description="Total number of sparse parameters in the node"
    )
    percent: Optional[float] = Field(
        None, description="Percentage of counts_sparse over counts"
    )

    @validator("*", pre=True)
    def validate_types(cls, value):
        return type_validator(value)

    @validator("percent", pre=True, always=True)
    def calculate_percent_if_none(cls, value, values):
        if value is None:
            counts = values.get("counts", 0)
            counts_sparse = values.get("counts_sparse", 0)
            return counts_sparse / counts if counts > 0 else 0.0
        return value
