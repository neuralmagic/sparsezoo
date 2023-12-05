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

from pydantic import BaseModel, Field, validator

from sparsezoo.analyze_v2.schemas.utils import type_validator


class DistributionAnalysisSchema(BaseModel):
    counts: Optional[int] = Field(..., description="Total number of parameters")
    mean: Optional[float]
    median: Optional[float]
    modes: Optional[List]
    sum_val: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    percentiles: Optional[Dict[float, float]]
    std_dev: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    entropy: Optional[float]
    bin_width: Optional[float]
    num_bins: Optional[int]
    hist: Optional[List[float]] = Field(
        ..., description="Frequency of the parameters, with respect to the bin edges"
    )
    bin_edges: Optional[List[float]] = Field(
        ..., description="Lower bound edges of each bin"
    )

    @validator("*", pre=True)
    def validate_types(cls, value):
        return type_validator(value)
