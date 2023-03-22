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

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


__all__ = [
    "NodeCounts",
    "NodeIO",
    "ZeroNonZeroParams",
    "DenseSparseOps",
    "ParameterSummary",
    "OpsSummary",
    "OperationSummary",
    "ParameterComponent",
]


class PropertyBaseModel(BaseModel):
    """
    https://github.com/samuelcolvin/pydantic/issues/935#issuecomment-1152457432

    Workaround for serializing properties with pydantic until
    https://github.com/samuelcolvin/pydantic/issues/935
    is solved
    """

    @classmethod
    def get_properties(cls):
        return [
            prop
            for prop in dir(cls)
            if isinstance(getattr(cls, prop), property)
            and prop not in ("__values__", "fields")
        ]

    def dict(
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,  # noqa: F821
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,  # noqa: F821
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> "DictStrAny":  # noqa: F821
        attribs = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        props = self.get_properties()
        # Include and exclude properties
        if include:
            props = [prop for prop in props if prop in include]
        if exclude:
            props = [prop for prop in props if prop not in exclude]

        # Update the attribute dict with the properties
        if props:
            attribs.update({prop: getattr(self, prop) for prop in props})

        return attribs


class NodeCounts(BaseModel):
    """
    Pydantic model for specifying the number zero and non-zero operations and the
    associated sparsity
    """

    total: int = Field(description="The total number of nodes")
    quantized: int = Field(description="The number of nodes that have been quantized")
    # quantizable
    pruned: int = Field(description="The number of nodes that have been pruned")
    prunable: int = Field(description="The number of nodes that can be pruned")


class NodeIO(BaseModel):
    """
    Pydantic model for the inputs and outputs of a node in the onnx model graph
    """

    name: str = Field(description="Name of the input/output in onnx model graph")
    shape: Optional[List[Union[None, int]]] = Field(
        description="Shape of the input/output in onnx model graph (assuming a "
        "batch size of 1)"
    )
    dtype: Optional[str] = Field(
        description="Data type of the values from the input/output"
    )


class ZeroNonZeroParams(PropertyBaseModel):
    """
    Pydantic model for specifying the number zero and non-zero operations and the
    associated sparsity
    """

    zero: int = Field(
        description="The number of parameters whose value is non-zero", default=0
    )
    non_zero: int = Field(
        description="The number of parameters whose value is zero", default=0
    )

    @property
    def sparsity(self):
        total_values = self.non_zero + self.zero
        if total_values > 0:
            return self.zero / total_values
        else:
            return 0


class DenseSparseOps(PropertyBaseModel):
    """
    Pydantic model for specifying the number dense and sparse operations and the
    associated operation sparsity
    """

    dense: int = Field(
        description="The number of operations performed during inference", default=0
    )
    sparse: int = Field(
        description="The number of operations that would have been performed "
        "during inference had it not been for model sparsity",
        default=0,
    )

    @property
    def sparsity(self):
        total_ops = self.sparse + self.dense
        if total_ops > 0:
            return self.sparse / (self.sparse + self.dense)
        else:
            return 0


class ParameterSummary(BaseModel):
    """
    Pydantic model for the analysis of the parameters of a node
    """

    total: int = Field(
        description="The number of parameters including those which have been pruned",
        default=0,
    )
    pruned: int = Field(
        description="The number of parameters that have been pruned", default=0
    )
    block_structure: Dict[str, ZeroNonZeroParams] = Field(
        description="The number of parameters when grouped into blocks", default={}
    )
    precision: Dict[str, ZeroNonZeroParams] = Field(
        description="The number of parameters at each precision level", default={}
    )


class OpsSummary(BaseModel):
    """
    Pydantic model for the analysis of a specific operation in a node, either ops
    or macs
    """

    total: int = Field(
        description="The total number of operations not accounting for sparsity"
    )
    pruned: int = Field(
        description="The number of operations not performed due to them being pruned"
    )
    block_structure: Dict[str, DenseSparseOps] = Field(
        description="The number of operations performed using each block grouping"
    )
    precision: Dict[str, DenseSparseOps] = Field(
        description="The number of operations performed at each precision"
    )


class OperationSummary(BaseModel):
    """
    Pydantic model for the analysis of the operations in a node
    """

    ops: OpsSummary = Field(
        description="The number of floating point or int operations"
    )
    macs: OpsSummary = Field(description="The number of multiply accumulates")


class ParameterComponent(BaseModel):
    """
    Pydantic model for the analysis of a parameter component of a node such as weight
    or bias
    """

    alias: str = Field(description="The type of parameter (weight, bias)")
    name: Optional[str] = Field(description="The name of the parameter")
    shape: Optional[List[Union[None, int]]] = Field(
        description="The shape of the parameter"
    )
    parameter_summary: ParameterSummary = Field(
        description="A summary of the parameter"
    )
    dtype: str = Field(description="The data type of the parameter")
