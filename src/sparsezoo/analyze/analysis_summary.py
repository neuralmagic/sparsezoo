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
Schema for generating model analysis summary
"""
import logging
import textwrap
from typing import List, Union

from pydantic import BaseModel

from sparsezoo.analyze.analysis import ModelAnalysis, YAMLSerializableBaseModel


_LOGGER = logging.getLogger(__name__)


class _SubtractBaseModel(BaseModel):
    """
    A BaseModel with subtraction support
    """

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


class Entry(_SubtractBaseModel):
    """
    The BaseModel representing a row entry

    :param sparsity: A float between 0-100 representing sparsity percentage
    :param quantized: A float between 0-100 representing quantized percentage
    """

    sparsity: float
    quantized: float

    _print_order = ["sparsity", "quantized"]

    def pretty_print(self, headers: bool = False):
        column_width = 40
        field_names = self._print_order
        field_values = []
        for field_name in field_names:
            field_value = getattr(self, field_name)
            if isinstance(field_value, float):
                field_value = f"{field_value:.2f}"
            field_values.append(textwrap.shorten(str(field_value), width=column_width))
        column_fmt = "{{:>{0}}} ".format(column_width)
        fmt_string = column_fmt * len(field_names)

        if headers:
            print(
                fmt_string.format(*(field_name.upper() for field_name in field_names))
            )

        print(fmt_string.format(*field_values))


class NamedEntry(Entry):
    """
    Entry with additional info like name, total and size
    """

    name: str
    total: float
    size: int

    _print_order = ["name", "total", "size"] + Entry._print_order


class TypedEntry(Entry):
    """
    Entry with additional info like type and size
    """

    type: str
    size: int

    _print_order = ["type", "size"] + Entry._print_order


class ModelEntry(Entry):
    """
    Entry which includes name of the model
    """

    model: str
    _print_order = ["model"] + Entry._print_order


class SizedModelEntry(ModelEntry):
    """
    A ModelEntry with additional info like count and size
    """

    count: int
    size: int
    _print_order = ModelEntry._print_order + ["count", "size"]


class Section(_SubtractBaseModel):
    """
    Represents a list of Entries with an optional name
    """

    entries: List[Union[NamedEntry, TypedEntry, SizedModelEntry, ModelEntry, Entry]]

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


class ModelAnalysisSummary(_SubtractBaseModel, YAMLSerializableBaseModel):
    sections: List[Section]

    def pretty_print(self):
        """
        Convenience function to pretty print ModelAnalysisSummary(...) objects
        """

        for section in self.sections:
            section.pretty_print()

    @classmethod
    def from_model_analysis(
        cls, analysis: ModelAnalysis, by_types: bool = False, by_layers: bool = False
    ) -> "ModelAnalysisSummary":
        """
        Factory method to generate a ModelAnalysisSummary object from a
        sparsezoo.ModelAnalysis object

        :param analysis: The ModelAnalysis object which the newly created
            ModelAnalysisSummary object will summarize
        :param by_types: flag to summarize analysis information by param and
            op type
        :param by_layers: flag to summarize analysis information by layers
        """
        sections = []

        if by_types:
            # TODO: Add analysis by_types section
            _LOGGER.info("analysis `by_types` is not implemented yet, will be ignored")

        if by_layers:
            # TODO: Add analysis by_layers section
            _LOGGER.info("analysis `by_layer` is not implemented yet, will be ignored")

        # Add Param analysis section
        param_section = Section(
            section_name="Params",
            entries=[
                SizedModelEntry(
                    model=analysis.model_name,
                    count=1234,
                    size=4321,
                    sparsity=90.453,
                    quantized=20.1,
                ),
            ],
        )

        # Add Ops analysis section
        ops_section = Section(
            section_name="Ops",
            entries=[
                SizedModelEntry(
                    model=analysis.model_name,
                    count=1234,
                    size=4321,
                    sparsity=90.453,
                    quantized=20.1,
                ),
            ],
        )

        # Add Overall model analysis section
        overall_section = Section(
            section_name="Overall",
            entries=[
                ModelEntry(
                    model=analysis.model_name,
                    sparsity=90.453,
                    quantized=20.1,
                )
            ],
        )

        sections = [param_section, ops_section, overall_section]
        return cls(sections=sections)


# local test

analysis = ModelAnalysis.create("/home/rahul/models/resnet50-dense.onnx")
summary = ModelAnalysisSummary.from_model_analysis(analysis)
summary.pretty_print()
