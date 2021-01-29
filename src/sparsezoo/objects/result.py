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
Code related to a model repo results
"""

from sparsezoo.objects.base import BaseObject


__all__ = ["Result"]


class Result(BaseObject):
    """
    A model repo metric result such as for performance, accuracy, etc

    :param result_id: the result id
    :param display_name: the display name for the result
    :param result_type: the result type e.g. benchmark or performance
    :param result_category: the result category e.g. cpu or gpu
    :param model_id: the model id of the model the result is for
    :param recorded_value: the recorded value of the result
    :param recorded_units: the units the recorded value was in
    :param recorded_format: any information of recorded format
    """

    def __init__(
        self,
        result_id: str,
        display_name: str,
        result_type: str,
        result_category: str,
        model_id: str,
        recorded_value: float,
        recorded_units: str,
        recorded_format: str,
        **kwargs,
    ):
        super(Result, self).__init__(**kwargs)
        self._result_id = result_id
        self._display_name = display_name
        self._result_type = result_type
        self._result_category = result_category
        self._model_id = model_id
        self._recorded_value = recorded_value
        self._recorded_units = recorded_units
        self._recorded_format = recorded_format

    @property
    def result_id(self) -> str:
        """
        :return: the result id
        """
        return self._result_id

    @property
    def model_id(self) -> str:
        """
        :return: the model id of the model the result is for
        """
        return self._model_id

    @property
    def display_name(self) -> str:
        """
        :return: the display name for the result
        """
        return self._display_name

    @property
    def result_type(self) -> str:
        """
        :return: the result type e.g. benchmark or performance
        """
        return self._result_type

    @property
    def result_category(self) -> str:
        """
        :return: the result category e.g. cpu or gpu
        """
        return self._result_category

    @property
    def recorded_value(self) -> float:
        """
        :return: the recorded value of the result
        """
        return self._recorded_value

    @property
    def recorded_units(self) -> str:
        """
        :return: the units the recorded value was in
        """
        return self._recorded_units

    @property
    def recorded_format(self) -> str:
        """
        :return: any information of recorded format
        """
        return self._recorded_format
