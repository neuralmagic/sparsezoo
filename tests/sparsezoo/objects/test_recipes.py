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

import tempfile

import pytest

from sparsezoo.model import Model


def test_recipe_getters():
    stub_with_multiple_recipes = "zoo:bert-base-wikipedia_bookcorpus-pruned90"
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
    model = Model(stub_with_multiple_recipes, temp_dir.name)

    default_recipe = model.recipes.default
    assert default_recipe.name == "recipe.md"

    all_recipes = model.recipes.recipes
    assert len(all_recipes) == 4

    recipe_name = "recipe_transfer_text_classification.md"
    found_by_name = model.recipes.get_recipe_by_name(recipe_name)
    assert found_by_name.name == recipe_name

    found_by_name = model.recipes.get_recipe_by_name("does_not_exist.md")
    assert found_by_name is None

    available_recipes = model.recipes.available
    assert len(available_recipes) == 4
    assert "recipe_transfer_token_classification" in available_recipes
    assert "recipe" in available_recipes


def test_custom_default():
    custom_default_name = "transfer_text_classification"
    stub_with_multiple_recipes = (
        "zoo:bert-base-wikipedia_bookcorpus-pruned90?recipe={}".format(
            custom_default_name
        )
    )
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
    model = Model(stub_with_multiple_recipes, temp_dir.name)

    expected_default_name = "recipe_" + custom_default_name + ".md"

    default_recipe = model.recipes.default
    assert default_recipe.name == expected_default_name

    available_recipes = model.recipes.available
    assert len(available_recipes) == 1
    assert available_recipes[0] == "recipe_transfer_text_classification"


def test_fail_default_on_empty():
    false_recipe_stub = "zoo:bert-base-wikipedia_bookcorpus-pruned90?recipe=nope"
    temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
    model = Model(false_recipe_stub, temp_dir.name)

    with pytest.raises(ValueError):
        _ = model.recipes.default
