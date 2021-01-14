"""
Code related to a model repo optimization file
"""

import logging
from enum import Enum

from sparsezoo.objects.file import File
from sparsezoo.objects.metadata import ModelMetadata


__all__ = ["OptimizationRecipeTypes", "OptimizationRecipe"]

_LOGGER = logging.getLogger(__name__)


class OptimizationRecipeTypes(Enum):
    """
    Types of recipes available in the sparsezoo
    """

    ORIGINAL = "original"
    TRANSFER_LEARN = "transfer_learn"


class OptimizationRecipe(File):
    """
    A model repo optimization recipe.

    :param model_metadata: the metadata for the model the file is for
    :param recipe_id: the recipe id
    :param recipe_type: the type of optimization recipe
    :param display_name: the display name for the optimization
    :param display_description: the display description for the optimization
    """

    def __init__(
        self,
        model_metadata: ModelMetadata,
        recipe_id: str,
        recipe_type: str,
        display_name: str,
        display_description: str,
        **kwargs,
    ):
        super(OptimizationRecipe, self).__init__(
            model_metadata=model_metadata, child_folder_name="recipes", **kwargs
        )
        self._recipe_id = recipe_id
        self._recipe_type = recipe_type
        self._display_name = display_name
        self._display_description = display_description

    @property
    def recipe_id(self) -> str:
        """
        :return: the optimization id
        """
        return self._recipe_id

    @property
    def recipe_type(self) -> str:
        """
        :return: the type of optimizations
        """
        return self._recipe_type

    @property
    def recipe_type_original(self) -> bool:
        """
        :return: True if this is the original optimization recipe
            that created the model, False otherwise
        """
        return self.recipe_type == OptimizationRecipeTypes.ORIGINAL.value

    @property
    def recipe_type_transfer_learn(self) -> bool:
        """
        :return: True if this is an optimization recipe for
            transfer learning from the created model, False otherwise
        """
        return self.recipe_type == OptimizationRecipeTypes.TRANSFER_LEARN.value

    @property
    def display_name(self):
        """
        :return: the display name for the optimization
        """
        return self._display_name

    @property
    def display_description(self) -> str:
        """
        :return: the display description for the optimization
        """
        return self._display_description
