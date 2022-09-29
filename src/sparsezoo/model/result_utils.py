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

from pydantic import BaseModel, Field


__all__ = ["ModelResult", "ValidationResult", "ThroughputResults"]


class ModelResult(BaseModel):
    """
    Base class to store common result information
    """

    result_type: str = Field(
        description="A string representing the type of "
        "result ex `training`, `inference`, etc"
    )
    recorded_value: float = Field(description="The float value of the result")
    recorded_units: str = Field(description="The unit in which result is specified")


class ValidationResult(ModelResult):
    """
    A class holding information for validation results
    """

    dataset_type: str = Field(
        description="A string representing the type of "
        "dataset used ex. `upstream`, `downstream`"
    )
    dataset_name: str = Field(
        description="The name of the dataset current " "result was measured on"
    )


class ThroughputResults(ModelResult):
    """
    A class holding information for throughput based results
    """

    device_info: str = Field(description="The device current result was measured on")
    num_cores: int = Field(
        description="Number of cores used while measuring " "this result"
    )
    batch_size: int = Field(
        description="The batch size used while measuring " "this result"
    )
