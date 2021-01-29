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
Code related to a model repo release version
"""

from sparsezoo.objects.base import BaseObject


__all__ = ["ReleaseVersion"]


class ReleaseVersion(BaseObject):
    """
    A model repo semantic release version. Will represent a version in the format

    MAJOR.MINOR.PATCH

    :param release_version_id: the release version id
    :param major_version: the major version of the release
    :param minor_version: the minor version of the release
    :param patch_version: the patch version of the release
    :param published: whether the release has been officially published
    """

    def __init__(
        self,
        release_version_id: str,
        major_version: str,
        minor_version: str,
        patch_version: str,
        published: bool,
        **kwargs,
    ):
        super(ReleaseVersion, self).__init__(**kwargs)
        self._release_version_id = release_version_id
        self._major_version = major_version
        self._minor_version = minor_version
        self._patch_version = patch_version
        self._published = published

    def __str__(self) -> str:
        """
        :return: The release version as semantic version string MAJOR.MINOR.PATCH
        """
        return f"{self.major_version}.{self.minor_version}.{self.patch_version}"

    @property
    def published(self) -> bool:
        """
        :return: whether the release has been officially published
        """
        return self._published

    @property
    def release_version_id(self) -> str:
        """
        :return: the release version id
        """
        return self._release_version_id

    @property
    def major_version(self) -> str:
        """
        :return: the major version of the release
        """
        return self._major_version

    @property
    def minor_version(self) -> str:
        """
        :return: the minor version of the release
        """
        return self._minor_version

    @property
    def patch_version(self) -> str:
        """
        :return: the patchversion of the release
        """
        return self._patch_version
