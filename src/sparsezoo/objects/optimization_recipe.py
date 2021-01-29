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
    :param display_description: the display description for the optimization
    """

    def __init__(
        self,
        model_metadata: ModelMetadata,
        recipe_id: str,
        recipe_type: str,
        display_description: str,
        **kwargs,
    ):
        super(OptimizationRecipe, self).__init__(
            model_metadata=model_metadata, child_folder_name="recipes", **kwargs
        )
        self._recipe_id = recipe_id
        self._recipe_type = recipe_type
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
