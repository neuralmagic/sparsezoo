"""
Code related to model repo tags
"""

from sparsezoo.objects.base import BaseObject


__all__ = ["Tag"]


class Tag(BaseObject):
    """
    A tag for a Model or OptimizationRecipe

    :param tag_id: the tag id
    :param display_name: the display name for tag
    :param model_id: the model id if the tag is associated with a Model
    :param recipe_id: the optimization id if tag is associated
        with an OptimizationRecipe
    :param name: the name for the tag
    """

    def __init__(
        self,
        tag_id: str,
        display_name: str,
        model_id: str,
        recipe_id: str,
        name: str,
        **kwargs,
    ):
        super(Tag, self).__init__(**kwargs)
        self._tag_id = tag_id
        self._display_name = display_name
        self._model_id = model_id
        self._recipe_id = recipe_id
        self._name = name

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
    def recipe_id(self) -> str:
        """
        :return: the optimization id if tag is associated with an OptimizationRecipe
        """
        return self._recipe_id
