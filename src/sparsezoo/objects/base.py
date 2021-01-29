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
Base objects for working with the sparsezoo
"""

from typing import Dict, List, Union


__all__ = ["BaseObject"]


class BaseObject:
    """
    A sparse zoo base object

    :param created: the date created
    :param modified: the date modified
    """

    def __init__(
        self,
        created: Union[str, None] = None,
        modified: Union[str, None] = None,
        **kwargs,
    ):
        self._created = created
        self._modified = modified

    @property
    def created(self) -> Union[str, None]:
        """
        :return: the date created
        """
        return self._created

    @property
    def modified(self) -> Union[str, None]:
        """
        :return: the date modifed
        """
        return self._modified

    def dict(self) -> Dict:
        """
        :return: The object as a dictionary
        """
        prop_dict = {}
        for prop in self._get_properties():
            if prop[0] == "_":
                prop = prop[1:]

            if not hasattr(self, prop):
                continue

            prop_value = getattr(self, prop)

            if isinstance(prop_value, BaseObject) or issubclass(
                type(prop_value), BaseObject
            ):
                prop_dict[prop] = prop_value.dict()
            elif isinstance(prop_value, list):
                prop_dict[prop] = [
                    elem.dict()
                    if isinstance(elem, BaseObject)
                    or issubclass(type(elem), BaseObject)
                    else elem
                    for elem in prop_value
                ]
            else:
                prop_dict[prop] = prop_value

        return prop_dict

    def _get_properties(self) -> List[str]:
        return list(vars(self).keys())
