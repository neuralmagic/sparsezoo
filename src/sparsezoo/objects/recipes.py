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
from typing import List, Union

from sparsezoo.objects import File


__all__ = ["Recipes"]

_LOGGER = logging.getLogger(__name__)


class Recipes:
    """
    Object to store a list of recipes for a downloaded model and pull the default

    :param recipes: list of recipes to store
    """

    _RECIPE_DEFAULT_NAME = "recipe.md"

    def __init__(self, recipes: Union[None, File, List[File]]):
        if recipes is None:
            recipes = []
        if isinstance(recipes, File):
            recipes = [recipes]

        self._recipes = recipes

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
            if recipe.name == self._RECIPE_DEFAULT_NAME:
                return recipe

        # fallback to first recipe in list
        _LOGGER.warning(
            "No default recipe {self._RECIPE_DEFAULT_NAME} found, falling back to"
            "first listed recipe"
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
