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
Code related to a downloadable interface
"""

import logging
import os
from typing import Union

from sparsezoo.utils import CACHE_DIR, create_dirs


__all__ = ["Downloadable"]

_LOGGER = logging.getLogger(__name__)


class Downloadable:
    """
    Downloadable interface with a default folder and file name

    :param folder_name: Name of the folder to save the downloads under
    :param override_parent_path: Path to override the default save path for where
        to save the folder and downloads at
    """

    def __init__(
        self,
        folder_name: str,
        override_parent_path: Union[str, None] = None,
        **kwargs,
    ):
        self._folder_name = folder_name
        self._override_parent_path = override_parent_path

    @property
    def folder_name(self) -> str:
        """
        :return: Name of the folder to save the downloads under
        """
        return self._folder_name

    @property
    def override_parent_path(self) -> str:
        """
        :return: Path to override the default save path for where to save
            the folder and downloads at
        """
        return self._override_parent_path

    @property
    def dir_path(self) -> str:
        """
        :return: The local path to download files into.
            Appends the folder_name to one of the following in order of resolution:
            [override_parent_path, SPARSEZOO_MODELS_PATH env variable,
            ~/.cache/sparszoo]
        """
        dir_path = self._override_parent_path

        if not dir_path:
            dir_path = os.getenv("SPARSEZOO_MODELS_PATH", "")

        if not dir_path:
            dir_path = CACHE_DIR

        dir_path = os.path.join(dir_path, self.folder_name)
        create_dirs(dir_path)

        return dir_path

    def download(
        self,
        overwrite: bool = False,
        refresh_token: bool = False,
        show_progress: bool = True,
    ):
        raise NotImplementedError()
