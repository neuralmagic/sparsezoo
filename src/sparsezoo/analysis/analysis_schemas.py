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

"""
A module that contains schema definitions for benchmarking and/or performance
analysis results
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


__all__ = [
    "NodeInferenceResult",
    "ImposedSparsificationInfo",
]


class NodeInferenceResult(BaseModel):
    """
    Schema definition for benchmark results for an onnx node
    """

    name: str = Field(description="The node's name")
    avg_run_time: PositiveFloat = Field(
        description="Average run time for current node in milli-secs",
    )
    extras: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra arguments for DeepSparse specific results",
    )


class ImposedSparsificationInfo(BaseModel):
    """
    Schema definition for applied sparsification techniques
    """

    sparsity: Optional[PositiveFloat] = Field(
        default=None,
        description="Globally imposed sparsity level, should be " "within (0, 1.0]",
    )

    sparsity_block_structure: Optional[str] = Field(
        default=None,
        description="The sparsity block structure applied to the onnx model;"
        " ex 2:4, 4",
    )

    quantization: bool = Field(
        default=False,
        description="Flag to ascertain if quantization should be applied or not",
    )

    recipe: Optional[str] = Field(
        default=None,
        description="The recipe to be applied",
    )


class BenchmarkSetup(BaseModel):
    batch_size: PositiveInt = Field(
        default=1,
        description="The batch size to use for benchmarking, defaults to 1",
    )

    num_cores: Optional[int] = Field(
        default=None,
        description="The number of cores to use for benchmarking, defaults "
        "to `None`, which represents all cores",
    )

    engine: str = Field(
        default="deepsparse",
        description="The engine to use for benchmarking, can be `deepsparse`"
        "or `onnxruntime`; defaults to `deepsparse`",
    )

    scenario: str = Field(
        default="sync",
        description="The scenario to use for benchmarking, could be `sync` or "
        "`async`; defaults to `sync`",
    )

    num_streams: Optional[int] = Field(
        default=None, description="Number of streams to use for benchmarking"
    )

    duration: Optional[int] = Field(
        default=None,
    )
