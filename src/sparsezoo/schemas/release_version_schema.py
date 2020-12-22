from typing import Dict
from sparsezoo.schemas.object_schema import ObjectSchema

__all__ = ["ReleaseVersionSchema"]


class ReleaseVersionSchema(ObjectSchema):
    def __init__(self, **kwargs):
        super(ReleaseVersionSchema, self).__init__(**kwargs)
        self._release_version_id = kwargs["release_version_id"]
        self._major_version = kwargs["major_version"]
        self._minor_version = kwargs["minor_version"]
        self._patch_version = kwargs["patch_version"]
        self._published = kwargs["published"]

    @property
    def published(self) -> bool:
        return self._published

    @property
    def release_version_id(self) -> str:
        return self._release_version_id

    @property
    def major_version(self) -> str:
        return self._major_version

    @property
    def minor_version(self) -> str:
        return self._minor_version

    @property
    def patch_version(self) -> str:
        return self._patch_version

    def dict(self) -> Dict:
        return {
            "release_version_id": self.release_version_id,
            "published": self.published,
            "major_version": self.major_version,
            "minor_version": self.minor_version,
            "patch_version": self.patch_version,
        }
