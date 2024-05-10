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

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from sparsezoo.analyze_v2.schemas.utils import type_validator


class DistributionAnalysisSchema(BaseModel):
    counts: Optional[int] = Field(..., description="Total number of parameters")
    mean: Optional[float] = None
    median: Optional[float] = None
    modes: Optional[List] = None
    sum_val: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    percentiles: Optional[Dict[float, float]] = None
    std_dev: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    entropy: Optional[float] = None
    bin_width: Optional[float] = None
    num_bins: Optional[int] = None
    hist: Optional[List[float]] = Field(
        ..., description="Frequency of the parameters, with respect to the bin edges"
    )
    bin_edges: Optional[List[float]] = Field(
        ..., description="Lower bound edges of each bin"
    )

    @field_validator("*", mode="before")
    @classmethod
    def validate_types(cls, value):
        return type_validator(value)
