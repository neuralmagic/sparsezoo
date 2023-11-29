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

from pydantic import BaseModel, validator

from sparsezoo.analyze_v2.model_validators.utils import type_validator


class QuantizationAnalysisModel(BaseModel):
    bits: float  # for val in tensor: total_bits += num_bits
    bits_quant: int
    percent: float
    # grouping: str  # Tensor, Channel

    @validator("*", pre=True)
    def validate_types(cls, value):
        return type_validator(value)
