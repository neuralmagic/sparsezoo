"""
Code related to model repo tags
"""

from sparsezoo.models.sparse_zoo_object import SparseZooObject


__all__ = ["Tag"]


class Tag(SparseZooObject):
    """
    A tag for a Model or OptimizationRecipe

    :param tag_id: the tag id
    :param display_name: the display name for tag
    :param model_id: the model id if the tag is associated with a Model
    :param optimization_id: the optimization id if tag is associated with an OptimizationRecipe
    :param name: the name for the tag
    """

    def __init__(self, **kwargs):
        super(Tag, self).__init__(**kwargs)
        self._tag_id = kwargs["tag_id"]
        self._display_name = kwargs["display_name"]
        self._model_id = kwargs["model_id"]
        self._optimization_id = kwargs["optimization_id"]
        self._name = kwargs["name"]

    @property
    def display_name(self) -> str:
        """
        :return: the display name for tag
        """
        return self._display_name

    @property
    def name(self) -> str:
        """
        :return: the name for the tag
        """
        return self._name

    @property
    def tag_id(self) -> str:
        """
        :return: the tag id
        """
        return self._tag_id

    @property
    def model_id(self) -> str:
        """
        :return: the model id of the model the tag is associated with
        """
        return self._model_id

    @property
    def optimization_id(self) -> str:
        """
        :return: the optimization id if tag is associated with an OptimizationRecipe
        """
        return self._optimization_id
