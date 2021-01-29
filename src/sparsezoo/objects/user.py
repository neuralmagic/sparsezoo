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
Code related to a model repo user
"""

from sparsezoo.objects.base import BaseObject


__all__ = ["User"]


class User(BaseObject):
    """
    A model repo user

    :param email: contact email
    :param name: name of user
    :param user_id: user id
    :param trusted: Whether the user is a trusted source
    """

    def __init__(
        self,
        email: str,
        name: str,
        user_id: str,
        trusted: bool,
        **kwargs,
    ):
        super(User, self).__init__(**kwargs)
        self._email = email
        self._name = name
        self._user_id = user_id
        self._trusted = trusted

    @property
    def email(self) -> str:
        """
        :return: contact email
        """
        return self._email

    @property
    def name(self) -> str:
        """
        :return: name of user
        """
        return self._name

    @property
    def user_id(self) -> str:
        """
        :return: user id
        """
        return self._user_id

    @property
    def trusted(self) -> bool:
        """
        :return: Whether the user is a trusted source
        """
        return self._trusted
