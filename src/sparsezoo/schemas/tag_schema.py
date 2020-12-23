"""
Code related to model repo tags
"""

from typing import Dict
from sparsezoo.schemas.object_schema import ObjectSchema


__all__ = ["RepoTag"]


class RepoTag(ObjectSchema):
    """
    A model repo tag

    :param tag_id: the tag id
    :param display_name: the display name for tag
    :param model_id: the model id of the model the tag is for
    :param name: the name for the ta
    """

    def __init__(self, **kwargs):
        super(RepoTag, self).__init__(**kwargs)
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
