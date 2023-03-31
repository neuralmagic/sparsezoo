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
from typing import List, Union

import yaml
from pydantic import BaseModel

from sparsezoo.analyze.analysis import YAMLSerializableBaseModel


class SubstractableBaseModel(BaseModel):
    def __sub__(self, other):
        my_fields = self.__fields__
        other_fields = other.__fields__

        assert list(my_fields) == list(other_fields)
        new_fields = {}
        for field in my_fields:
            my_value = getattr(self, field)
            other_value = getattr(other, field)

            assert type(my_value) == type(other_value)
            if isinstance(my_value, str):
                new_fields[field] = f"{my_value} - {other_value}"
            elif isinstance(my_value, list):
                new_fields[field] = [
                    item_a - item_b for item_a, item_b in zip(my_value, other_value)
                ]
            else:
                new_fields[field] = my_value - other_value

        return self.__class__(**new_fields)


class Entry(SubstractableBaseModel):
    sparsity: float
    quantized: float


class NamedEntry(Entry):
    name: str
    total: float
    size: int


class TypedEntry(Entry):
    type: str
    size: int


class ModelEntry(Entry):
    model: str


class SizedModelEntry(ModelEntry):
    count: int
    size: int


class Section(SubstractableBaseModel):
    entries: List[Union[NamedEntry, TypedEntry, SizedModelEntry, ModelEntry, Entry]]


class ModelAnalysisSummary(SubstractableBaseModel, YAMLSerializableBaseModel):
    sections: List[Section]
