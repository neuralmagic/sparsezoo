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

from sparsezoo.analyze_v2.schemas.utils import type_validator


class QuantizationSummaryAnalysisSchema(BaseModel):
    bits: float = Field(..., description="Total bits required to store the weights")
    bits_quant: int = Field(
        ...,
        description=(
            "Total quantized bits required to store the weights."
            "Here we assume if the layer is quantized, the entire array is quantized"
        ),
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
            bits = values.get("bits", 0)
            bits_quant = values.get("bits_quant", 0)
            return bits_quant / bits if bits > 0 else 0.0
        return value

    def __add__(self, model: BaseModel):
        validator_model = None
        if isinstance(model, QuantizationSummaryAnalysisSchema):
            validator_model = QuantizationSummaryAnalysisSchema

        if validator_model is not None:
            return validator_model(
                bits=self.bits + model.bits,
                bits_quant=self.bits_quant + model.bits_quant,
            )


class QuantizationAnalysisSchema(QuantizationSummaryAnalysisSchema):
    grouping: str = Field(..., description="The combining group name")

    def __add__(self, model: BaseModel):
        validator_model = None
        if isinstance(model, QuantizationAnalysisSchema):
            validator_model = QuantizationAnalysisSchema

        if validator_model is not None and self.grouping == model.grouping:
            return validator_model(
                grouping=self.grouping,
                bits=self.bits + model.bits,
                bits_quant=self.bits_quant + model.bits_quant,
            )
