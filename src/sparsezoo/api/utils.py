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

from typing import Any, Callable, Dict, List


def to_camel_case(string: str):
    "Convert string to camel case"
    components = string.split("_")
    return components[0] + "".join(word.title() for word in components[1:])


def to_snake_case(string: str):
    "Convert string to snake case"
    return "".join(
        [
            "_" + character.lower() if character.isupper() else character
            for character in string
        ]
    ).lstrip("_")


def map_keys(
    dictionary: Dict[str, Any], mapper: Callable[[str], str]
) -> Dict[str, str]:
    """
    Given a dictionary, update its keys to a given mapper callable.

    If the value of the dict is a List of Dict or Dict of Dict, recursively map
    its keys
    """
    mapped_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, List) or isinstance(value, Dict):
            value_type = type(value)
            mapped_dict[mapper(key)] = value_type(
                map_keys(dictionary=sub_dict, mapper=mapper) for sub_dict in value
            )
        else:
            mapped_dict[mapper(key)] = value

    return mapped_dict
