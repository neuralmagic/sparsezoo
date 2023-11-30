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

from pydantic import BaseModel, Field, validator

from sparsezoo.analyze_v2.model_validators.utils import type_validator


class QuantizationAnalysisModel(BaseModel):
    grouping: str = Field(..., description="The combining group name")
    bits: float = Field(..., description="Total bits required to store the weights")
    bits_quant: int = Field(
        ...,
        description=(
            "Total quantized bits required to store the weights."
            "Here we assume if the layer is quantized, the entire array is quantized"
        ),
    )
    percent: float = Field(..., description="Percentage of bits_quant over bits")

    @validator("*", pre=True)
    def validate_types(cls, value):
        return type_validator(value)
