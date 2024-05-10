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
import textwrap
from typing import ClassVar, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, computed_field


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
PrintOrderType = ClassVar[List[str]]


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
        None,
        description="Shape of the input/output in onnx model graph (assuming a "
        "batch size of 1)",
    )
    dtype: Optional[str] = Field(
        None, description="Data type of the values from the input/output"
    )


class ZeroNonZeroParams(BaseModel):
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

    @computed_field(repr=True, return_type=Union[int, float])
    @property
    def sparsity(self):
        total_values = self.total
        if total_values > 0:
            return self.zero / total_values
        else:
            return 0.0

    @computed_field(repr=True, return_type=int)
    @property
    def total(self):
        return self.non_zero + self.zero


class DenseSparseOps(BaseModel):
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

    @computed_field(repr=True, return_type=Union[int, float])
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
    name: Optional[str] = Field(None, description="The name of the parameter")
    shape: Optional[List[Union[None, int]]] = Field(
        None, description="The shape of the parameter"
    )
    parameter_summary: ParameterSummary = Field(
        description="A summary of the parameter"
    )
    dtype: str = Field(description="The data type of the parameter")


class Entry(BaseModel):
    """
    A BaseModel with subtraction and pretty_print support
    """

    _print_order: PrintOrderType = []

    def __sub__(self, other):
        """
        Allows base functionality for all inheriting classes to be subtract-able,
        subtracts the fields of self with other while providing some additional
        support for string and unrolling list type fields
        """
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

            assert type(my_value) is type(other_value)
            if field == "section_name":
                new_fields[field] = my_value
            elif isinstance(my_value, str):
                new_fields[field] = (
                    my_value
                    if my_value == other_value
                    else f"{my_value} - {other_value}"
                )
            elif isinstance(my_value, list):
                new_fields[field] = [
                    item_a - item_b for item_a, item_b in zip(my_value, other_value)
                ]
            else:
                new_fields[field] = my_value - other_value

        return self.__class__(**new_fields)

    def pretty_print(self, headers: bool = False, column_width=30):
        """
        pretty print current Entry object with all it's fields
        """
        field_names = self._print_order
        field_values = []
        for field_name in field_names:
            field_value = getattr(self, field_name)
            if isinstance(field_value, float):
                field_value = f"{field_value:.2f}"
            field_values.append(field_value)

        if headers:
            print(
                multiline_pretty_print(
                    row=[field_name.upper() for field_name in field_names],
                    column_width=column_width,
                )
            )
        print(multiline_pretty_print(row=field_values, column_width=column_width))


class BaseEntry(Entry):
    """
    The BaseModel representing a row entry

    :param sparsity: A float between 0-100 representing sparsity percentage
    :param quantized: A float between 0-100 representing quantized percentage
    """

    sparsity: float
    quantized: float

    _print_order: PrintOrderType = ["sparsity", "quantized"]


class NamedEntry(BaseEntry):
    """
    BaseEntry with additional info like name, total and size
    """

    name: str
    total: float
    size: int

    _print_order: PrintOrderType = ["name", "total", "size"] + BaseEntry._print_order


class TypedEntry(BaseEntry):
    """
    BaseEntry with additional info like type and size
    """

    type: str
    size: int

    _print_order: PrintOrderType = ["type", "size"] + BaseEntry._print_order


class ModelEntry(BaseEntry):
    """
    BaseEntry which includes name of the model
    """

    model: str
    _print_order: PrintOrderType = ["model"] + BaseEntry._print_order


class SizedModelEntry(ModelEntry):
    """
    A ModelEntry with additional info like count and size
    """

    count: int
    size: Union[int, float]
    _print_order: PrintOrderType = ModelEntry._print_order + ["count", "size"]


class PerformanceEntry(BaseEntry):
    """
    A BaseEntry with additional performance info
    """

    model: str
    latency: float
    throughput: float
    supported_graph: float

    _print_order: PrintOrderType = [
        "model",
        "latency",
        "throughput",
        "supported_graph",
    ] + BaseEntry._print_order


class NodeTimingEntry(Entry):
    """
    A BaseEntry with additional performance info
    """

    node_name: str
    avg_runtime: float

    _print_order: PrintOrderType = [
        "node_name",
        "avg_runtime",
    ] + Entry._print_order


class Section(Entry):
    """
    Represents a list of Entries with an optional name
    """

    entries: List[
        Union[
            NodeTimingEntry,
            PerformanceEntry,
            NamedEntry,
            TypedEntry,
            SizedModelEntry,
            ModelEntry,
            BaseEntry,
        ]
    ]

    section_name: str = ""

    def pretty_print(self):
        """
        pretty print current section, with its entries
        """
        if self.section_name:
            if not self.entries:
                print(f"No entries found in: {self.section_name}")
            else:
                print(f"{self.section_name}:")

        for idx, entry in enumerate(self.entries):
            if idx == 0:
                entry.pretty_print(headers=True)
            else:
                entry.pretty_print(headers=False)
        print()

    def __sub__(self, other: "Section"):
        """
        A method that allows us to subtract two Section objects,
        If the section includes `NamedEntry` or `TypedEntry` then we only compare
        the entries which have the same name or type (and others will be ignored),
        Subtraction of other Entry types is delegated to their own implementation
        This function also assumes that a Section has entries of the same type
        """

        if not isinstance(other, Section):
            raise TypeError(
                f"unsupported operand type(s) for -: {type(self)} and {type(other)}"
            )

        section_name = self.section_name or ""
        self_entries, other_entries = self.get_comparable_entries(other)

        compared_entries = [
            self_entry - other_entry
            for self_entry, other_entry in zip(self_entries, other_entries)
        ]

        return Section(
            section_name=section_name,
            entries=compared_entries,
        )

    def get_comparable_entries(self, other: "Section") -> Tuple[List[Entry], ...]:
        """
        Get comparable entries by same name or type if they belong to
        `NamedEntry`, `TypedEntry`, or `NodeTimingEntry`, else return all entries

        :return: A tuple composed of two lists, containing comparable entries
            in correct order from current and other Section objects
        """
        assert self.entries
        entry_type_to_extractor = {
            "NamedEntry": lambda entry: entry.name,
            "TypedEntry": lambda entry: entry.type,
            "NodeTimingEntry": lambda entry: entry.node_name,
        }
        entry_type = self.entries[0].__class__.__name__

        if entry_type not in entry_type_to_extractor:
            return self.entries, other.entries

        key_extractor = entry_type_to_extractor[entry_type]
        self_entry_dict = {key_extractor(entry): entry for entry in self.entries}
        other_entry_dict = {key_extractor(entry): entry for entry in other.entries}

        self_comparable_entries = []
        other_comparable_entries = []

        for key, value in self_entry_dict.items():
            if key in other_entry_dict:
                self_comparable_entries.append(value)
                other_comparable_entries.append(other_entry_dict[key])

        if len(self_comparable_entries) != len(self_entry_dict):
            _LOGGER.info(
                "Found mismatching entries, these will be ignored during "
                f"comparison in Section: {self.section_name}"
            )
        return self_comparable_entries, other_comparable_entries


def multiline_pretty_print(row: List[str], column_width=20) -> str:
    """
    Formats the contents of the specified row into a multiline string which
    each column is wrapped into a multiline string if its length is greater
    than the specified column_width

    :param row: A list of strings to be formatted into a multiline row
    :param column_width: The max width of each column for formatting, default is 20
    :returns: A multiline formatted string representing the row,
    """
    row = [str(column) for column in row]
    result_string = ""
    col_delim = " "
    wrapped_row = [textwrap.wrap(col, column_width) for col in row]
    max_lines_needed = max(len(col) for col in wrapped_row)

    for line_idx in range(max_lines_needed):
        result_string += col_delim
        for column in wrapped_row:
            if line_idx < len(column):
                result_string += column[line_idx].ljust(column_width)
            else:
                result_string += " " * column_width
            result_string += col_delim
        if line_idx < max_lines_needed - 1:
            result_string += "\n"
    return result_string
