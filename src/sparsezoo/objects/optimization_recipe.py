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
import os
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from sparsezoo.objects.model import Model

from enum import Enum

from sparsezoo.models.zoo import Zoo
from sparsezoo.objects.file import File
from sparsezoo.objects.metadata import ModelMetadata


__all__ = ["OptimizationRecipeTypes", "OptimizationRecipe"]


_LOGGER = logging.getLogger(__name__)


@Zoo.register("OptimizationRecipeTypes")
class OptimizationRecipeTypes(Enum):
    """
    Types of recipes available in the sparsezoo
    """

    ORIGINAL = "original"
    TRANSFER_LEARN = "transfer_learn"


class OptimizationRecipe(File):
    """
    A model repo recipe.

    :param model_metadata: the metadata for the model the file is for
    :param recipe_id: the recipe id
    :param recipe_type: the type of optimization recipe
    :param display_description: the display description for the optimization
    :param base_stub: the stub for the base model of this recipe, if any
    """

    def __init__(
        self,
        model_metadata: ModelMetadata,
        recipe_id: str,
        recipe_type: str,
        display_description: str,
        base_stub: Optional[str],
        **kwargs,
    ):
        super(OptimizationRecipe, self).__init__(
            model_metadata=model_metadata, child_folder_name="recipes", **kwargs
        )
        self._recipe_id = recipe_id
        self._recipe_type = recipe_type
        self._display_description = display_description
        self._base_stub = base_stub

    @staticmethod
    @Zoo.register("OptimizationRecipe")
    def construct(
        model_metadata: ModelMetadata,
        recipe_id: str,
        recipe_type: str,
        display_description: str,
        base_stub: Optional[str],
        **kwargs,
    ):
        return OptimizationRecipe(
            model_metadata=model_metadata,
            recipe_id=recipe_id,
            recipe_type=recipe_type,
            display_description=display_description,
            base_stub=base_stub,
            **kwargs,
        )

    @property
    def recipe_id(self) -> str:
        """
        :return: the recipe id
        """
        return self._recipe_id

    @property
    def recipe_type(self) -> str:
        """
        :return: the type of recipe
        """
        return self._recipe_type

    @property
    def recipe_type_original(self) -> bool:
        """
        :return: True if this is the original recipe that created the
            model, False otherwise
        """
        return self.recipe_type == OptimizationRecipeTypes.ORIGINAL.value

    @property
    def recipe_type_transfer_learn(self) -> bool:
        """
        :return: True if this is a recipe for transfer learning from the
            created model, False otherwise
        """
        return self.recipe_type == OptimizationRecipeTypes.TRANSFER_LEARN.value

    @property
    def display_name(self):
        """
        :return: the display name for the recipe
        """
        return self._display_name

    @property
    def display_description(self) -> str:
        """
        :return: the display description for the recipe
        """
        return self._display_description

    @property
    def base_stub(self) -> Optional[str]:
        """
        :return: the stub for the base model of this recipe, if any
        """
        return self._base_stub

    @property
    def stub(self) -> str:
        """
        :return: full path for where the recipe is located in the sparsezoo
        """
        return self.model_metadata.stub

    def load_model(self) -> "Model":
        """
        :return: the model associated with the recipe
        """
        return Zoo.load_model_from_recipe(
            recipe=self,
            override_folder_name=os.path.dirname(self.folder_name),
            override_parent_path=self.override_parent_path,
        )
