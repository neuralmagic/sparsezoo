from typing import Dict
from sparsezoo.schemas.object_schema import ObjectSchema


__all__ = ["TagSchema"]


class TagSchema(ObjectSchema):
    def __init__(self, **kwargs):
        super(TagSchema, self).__init__(**kwargs)
        self._tag_id = kwargs["tag_id"]
        self._display_name = kwargs["display_name"]
        self._model_id = kwargs["model_id"]
        self._name = kwargs["name"]

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def tag_id(self) -> str:
        return self._tag_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "tag_id": self.tag_id,
            "name": self.name,
            "display_name": self.display_name,
        }
