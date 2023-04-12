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
import logging
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

_LOGGER = logging.getLogger(__name__)


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
        total_values = self.total
        if total_values > 0:
            return self.zero / total_values
        else:
            return 0

    @property
    def total(self):
        return self.non_zero + self.zero


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


class Entry(BaseModel):
    """
    A BaseModel with subtraction and pretty_print support
    """

    _print_order: List[str] = []

    def __sub__(self, other):
        my_fields = self.__fields__
        other_fields = other.__fields__

        assert list(my_fields) == list(other_fields)
        new_fields = {}
        for field in my_fields:
            if field.startswith("_"):
                # ignore private fields
                continue
            my_value = getattr(self, field)
            other_value = getattr(other, field)

            assert type(my_value) == type(other_value)
            if field == "section_name":
                new_fields[field] = my_value
            elif isinstance(my_value, str):
                new_fields[field] = f"{my_value} - {other_value}"
            elif isinstance(my_value, list):
                new_fields[field] = [
                    item_a - item_b for item_a, item_b in zip(my_value, other_value)
                ]
            else:
                new_fields[field] = my_value - other_value

        return self.__class__(**new_fields)

    def pretty_print(self, headers: bool = False):
        column_width = 15
        field_names = self._print_order
        field_values = []
        for field_name in field_names:
            field_value = getattr(self, field_name)
            if isinstance(field_value, float):
                field_value = f"{field_value:.2f}"
            if field_name == "model":
                field_value = field_value[-40:]
            field_values.append(field_value)

        column_fmt = "{{:>{0}}} ".format(column_width)
        fmt_string = "{:>40}" + (column_fmt * (len(field_names) - 1))

        if headers:
            print(
                fmt_string.format(*(field_name.upper() for field_name in field_names))
            )

        print(fmt_string.format(*field_values))


class BaseEntry(Entry):
    """
    The BaseModel representing a row entry

    :param sparsity: A float between 0-100 representing sparsity percentage
    :param quantized: A float between 0-100 representing quantized percentage
    """

    sparsity: float
    quantized: float

    _print_order = ["sparsity", "quantized"]


class NamedEntry(BaseEntry):
    """
    BaseEntry with additional info like name, total and size
    """

    name: str
    total: float
    size: int

    _print_order = ["name", "total", "size"] + BaseEntry._print_order


class TypedEntry(BaseEntry):
    """
    BaseEntry with additional info like type and size
    """

    type: str
    size: int

    _print_order = ["type", "size"] + BaseEntry._print_order


class ModelEntry(BaseEntry):
    """
    BaseEntry which includes name of the model
    """

    model: str
    _print_order = ["model"] + BaseEntry._print_order


class SizedModelEntry(ModelEntry):
    """
    A ModelEntry with additional info like count and size
    """

    count: int
    size: int
    _print_order = ModelEntry._print_order + ["count", "size"]


class PerformanceEntry(BaseEntry):
    """
    A BaseEntry with additional performance info
    """

    model: str
    latency: float
    throughput: float
    supported_graph: float

    _print_order = [
        "model",
        "latency",
        "throughput",
        "supported_graph",
    ] + BaseEntry._print_order


class Section(Entry):
    """
    Represents a list of Entries with an optional name
    """

    entries: List[Union[NamedEntry, TypedEntry, SizedModelEntry, ModelEntry, BaseEntry]]

    section_name: str = ""

    def pretty_print(self):
        if self.section_name:
            print(f"{self.section_name}:")

        for idx, entry in enumerate(self.entries):
            if idx == 0:
                entry.pretty_print(headers=True)
            else:
                entry.pretty_print(headers=False)
        print()

    def __sub__(self, other):
        """
        A method that allows us to subtract two Section objects,
        If the section includes `NamedEntry` or `TypedEntry` then we only compare
        the entries which have the same name or type (and others will be ignored),
        Subtraction of other Entry types is delegated to their own implementation
        This function also assumes that a Section has entries of the same type
        """
        section_name = self.section_name or ""
        self_entries, other_entries = self._get_entries_to_compare(other)

        compared_entries = [
            self_entry - other_entry
            for self_entry, other_entry in zip(self_entries, other_entries)
        ]

        return Section(
            section_name=section_name,
            entries=compared_entries,
        )

    def _get_entries_to_compare(self, other):

        assert self.entries
        entry_type_to_comparator = {
            "NamedEntry": lambda obj: obj.name,
            "TypedEntry": lambda obj: obj.type,
        }

        self_entry_dict = {
            entry_type_to_comparator[entry.__class__.__name__](entry): entry
            for entry in self.entries
            if entry.__class__.__name__ in entry_type_to_comparator
        }
        other_entry_dict = {
            entry_type_to_comparator[entry.__class__.__name__](entry): entry
            for entry in self.entries
            if entry.__class__.__name__ in entry_type_to_comparator
        }

        self_comparable_entries = []
        other_comparable_entries = []

        for key, value in self_entry_dict.items():
            if key in other_entry_dict:
                self_comparable_entries.append(value)
                other_comparable_entries.append(other_entry_dict[key])

        if self_comparable_entries:
            return self_comparable_entries, other_comparable_entries

        return self.entries, other.entries
