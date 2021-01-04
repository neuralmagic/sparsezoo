"""
Code related to a model repo release version
"""
from sparsezoo.models.sparse_zoo_object import SparseZooObject


__all__ = ["ReleaseVersion"]


class ReleaseVersion(SparseZooObject):
    """
    A model repo semantic release verison. Will represent a version in the format

    MAJOR.MINOR.PATCH

    :param release_version_id: the release version id
    :param major_version: the major version of the release
    :param minor_version: the minor version of the release
    :param patch_version: the patch version of the release
    :param published: whether the release has been officially published
    """

    def __init__(self, **kwargs):
        super(ReleaseVersion, self).__init__(**kwargs)
        self._release_version_id = kwargs["release_version_id"]
        self._major_version = kwargs["major_version"]
        self._minor_version = kwargs["minor_version"]
        self._patch_version = kwargs["patch_version"]
        self._published = kwargs["published"]

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

    def __str__(self) -> str:
        """
        :return: The release version as semantic version string MAJOR.MINOR.PATCH
        """
        return f"{self.major_version}.{self.minor_version}.{self.patch_version}"
