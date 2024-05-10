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

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from sparsezoo.objects import File


__all__ = ["Recipes"]

_LOGGER = logging.getLogger(__name__)


class Recipes:
    """
    Object to store a list of recipes for a downloaded model and pull the default

    :param recipes: list of recipes to store
    :param stub_params: dictionary that may contain custom default recipes names
    """

    _RECIPE_DEFAULT_NAME = "recipe.md"

    def __init__(
        self,
        recipes: Optional[Union[File, List[File]]],
        stub_params: Dict[str, str] = {},
    ):
        if recipes is None:
            recipes = []
        if isinstance(recipes, File):
            recipes = [recipes]
        self._recipes = recipes

        self._default_recipe_name = self._RECIPE_DEFAULT_NAME
        custom_default = stub_params.get("recipe_type") or stub_params.get("recipe")
        if custom_default is not None:
            self._default_recipe_name = "recipe_" + custom_default

    @property
    def available(self) -> Optional[List[str]]:
        """
        :return: List of all recipe names, or None if none are available
        """
        if len(self._recipes) == 0:
            return None
        return [Path(recipe.name).stem for recipe in self._recipes]

    @property
    def recipes(self) -> List:
        """
        :return: The full list of recipes
        """
        return self._recipes

    @property
    def default(self) -> File:
        """
        :return: The default recipe in the recipe list
        """
        for recipe in self._recipes:
            if recipe.name.startswith(self._default_recipe_name):
                return recipe

        # fallback to first recipe in list
        if len(self._recipes) == 0:
            raise ValueError("No recipes found, could not retrieve a default.")

        _LOGGER.warning(
            f"No default recipe {self._default_recipe_name} found, falling back to "
            f"first listed recipe {self._recipes[0].name}"
        )
        return self._recipes[0]

    def get_recipe_by_name(self, recipe_name: str) -> Union[File, None]:
        """
        Returns the File for the recipe matching the name recipe_name if it exists

        :param recipe_name: recipe filename to search for
        :return: File with the name recipe_name, or None if it doesn't exist
        """

        for recipe in self._recipes:
            if recipe.name == recipe_name:
                return recipe

        return None  # no matching recipe found
