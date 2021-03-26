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

import os
import shutil

import pytest

from sparsezoo import Zoo
from sparsezoo.utils import CACHE_DIR
from tests.sparsezoo.utils import validate_downloaded_model


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {
                "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
                "override_folder_name": "test_folder",
            },
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
        ),
    ],
)
def test_load_model(model_args, other_args):
    model = Zoo.load_model(**model_args, **other_args)
    model.download(overwrite=True)
    validate_downloaded_model(model, model_args, other_args)
    shutil.rmtree(model.dir_path)


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {
                "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
                "override_folder_name": "test_folder",
            },
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
        ),
    ],
)
def test_download_model(model_args, other_args):
    model = Zoo.download_model(**model_args, **other_args)
    validate_downloaded_model(model, model_args, other_args)
    shutil.rmtree(model.dir_path)


@pytest.mark.parametrize(
    "stub, model_args, other_args",
    [
        [
            (
                "cv/classification/mobilenet_v1-1.0/pytorch/"
                "sparseml/imagenet/pruned-conservative"
            ),
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {},
        ]
    ],
)
def test_load_model_from_stub(stub, model_args, other_args):
    model = Zoo.load_model_from_stub(stub, **other_args)
    model.download(overwrite=True)
    for key in model_args:
        if key and hasattr(model, key):
            assert getattr(model, key) == model_args[key]
    shutil.rmtree(model.dir_path)


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        ({"domain": "cv", "sub_domain": "classification"}, {}),
        ({"domain": "cv", "sub_domain": "classification"}, {"page_length": 1}),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
            },
            {},
        ),
        ({"domain": "cv", "sub_domain": "classification", "optim_name": "base"}, {}),
    ],
)
def test_search_models(model_args, other_args):
    models = Zoo.search_models(**model_args, **other_args)

    for model in models:
        for key, value in model_args.items():
            assert getattr(model, key) == value

    if "page_length" in other_args:
        assert len(models) <= other_args["page_length"]


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
        ),
    ],
)
def test_search_similar_models(model_args, other_args):
    model = Zoo.load_model(**model_args, **other_args)
    similar = Zoo.search_optimized_models(model)
    assert len(similar) > 0

    for sim in similar:
        assert sim
        assert sim.domain == model.domain
        assert sim.sub_domain == model.sub_domain
        assert sim.architecture == model.architecture
        assert sim.sub_architecture == model.sub_architecture


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
        ),
    ],
)
def test_search_optimized_models(model_args, other_args):
    model = Zoo.load_model(**model_args, **other_args)
    optimized = Zoo.search_optimized_models(model)
    assert len(optimized) > 0

    for sim in optimized:
        assert sim
        assert sim.domain == model.domain
        assert sim.sub_domain == model.sub_domain
        assert sim.architecture == model.architecture
        assert sim.sub_architecture == model.sub_architecture
        assert sim.framework == model.framework
        assert sim.repo == model.repo
        assert sim.dataset == model.dataset
        assert sim.training_scheme == model.training_scheme


@pytest.mark.parametrize(
    "recipe_args,other_args",
    [
        ({"domain": "cv", "sub_domain": "classification"}, {}),
        ({"domain": "cv", "sub_domain": "classification"}, {"page_length": 1}),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
            },
            {},
        ),
        ({"domain": "cv", "sub_domain": "classification", "optim_name": "pruned"}, {}),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "recipe_type": "transfer_learn",
            },
            {},
        ),
        (
            {"domain": "cv", "sub_domain": "classification", "recipe_type": "original"},
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "recipe_type": "transfer_learn",
            },
            {},
        ),
    ],
)
def test_search_recipes(recipe_args, other_args):
    recipes = Zoo.search_recipes(**recipe_args, **other_args)

    for recipe in recipes:
        for key, value in recipe_args.items():
            if key == "recipe_type":
                assert getattr(recipe, key) == value
            else:
                assert getattr(recipe.model_metadata, key) == value

    if "page_length" in other_args:
        assert len(recipes) <= other_args["page_length"]


@pytest.mark.parametrize(
    "model_args,other_args,other_recipe_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
            {"recipe_type": "original"},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet_v1",
                "sub_architecture": "50",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
            {"recipe_type": "transfer_learn"},
        ),
    ],
)
def test_search_optimized_recipes(model_args, other_args, other_recipe_args):
    model = Zoo.load_model(**model_args, **other_args)
    recipes = Zoo.search_optimized_recipes(model, **other_recipe_args)
    assert len(recipes) > 0

    for recipe in recipes:
        assert recipe
        assert recipe.model_metadata.domain == model.domain
        assert recipe.model_metadata.sub_domain == model.sub_domain
        assert recipe.model_metadata.architecture == model.architecture
        assert recipe.model_metadata.sub_architecture == model.sub_architecture
        assert recipe.model_metadata.framework == model.framework
        assert recipe.model_metadata.repo == model.repo
        assert recipe.model_metadata.dataset == model.dataset
        assert recipe.model_metadata.training_scheme == model.training_scheme

        if "recipe_type" in other_recipe_args:
            assert recipe.recipe_type == other_recipe_args["recipe_type"]


@pytest.mark.parametrize(
    "model_stub,other_args",
    [
        (
            "cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none",
            {},
        ),
    ],
)
def test_search_optimized_recipes_from_stub(model_stub, other_args):
    model = Zoo.load_model_from_stub(model_stub, **other_args)
    recipes = Zoo.search_optimized_recipes(model_stub)
    assert len(recipes) > 0

    for recipe in recipes:
        assert recipe
        assert recipe.model_metadata.domain == model.domain
        assert recipe.model_metadata.sub_domain == model.sub_domain
        assert recipe.model_metadata.architecture == model.architecture
        assert recipe.model_metadata.sub_architecture == model.sub_architecture
        assert recipe.model_metadata.framework == model.framework
        assert recipe.model_metadata.repo == model.repo
        assert recipe.model_metadata.dataset == model.dataset
        assert recipe.model_metadata.training_scheme == model.training_scheme


@pytest.mark.parametrize(
    "recipe_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {
                "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
                "override_folder_name": "test_folder",
            },
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet_v1",
                "sub_architecture": "50",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned_quant",
                "optim_category": "moderate",
                "optim_target": None,
                "recipe_type": "transfer_learn",
            },
            {},
        ),
    ],
)
def test_load_recipe(recipe_args, other_args):
    recipe = Zoo.load_recipe(**recipe_args, **other_args)
    recipe.download(overwrite=True)
    assert os.path.exists(recipe.path)
    shutil.rmtree(recipe.dir_path)


@pytest.mark.parametrize(
    "recipe_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {
                "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
                "override_folder_name": "test_folder",
            },
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet_v1",
                "sub_architecture": "50",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned_quant",
                "optim_category": "moderate",
                "optim_target": None,
                "recipe_type": "transfer_learn",
            },
            {},
        ),
    ],
)
def test_download_recipe(recipe_args, other_args):
    recipe = Zoo.download_recipe(**recipe_args, **other_args)
    assert os.path.exists(recipe.path)
    shutil.rmtree(recipe.dir_path)


@pytest.mark.parametrize(
    "recipe_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {
                "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
                "override_folder_name": "test_folder",
            },
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet_v1",
                "sub_architecture": "50",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned_quant",
                "optim_category": "moderate",
                "optim_target": None,
                "recipe_type": "transfer_learn",
            },
            {},
        ),
    ],
)
def test_load_model_from_recipe(recipe_args, other_args):
    recipe = Zoo.load_recipe(**recipe_args, **other_args)
    recipe_model = Zoo.load_model_from_recipe(recipe, **other_args)
    model_dict = recipe_model.dict()
    for field, value in recipe.model_metadata.dict().items():
        # TODO temporary fix while model apis need to be updated
        if field == "created" or field == "modified" or field == "release_version":
            continue
        assert model_dict[field] == value


@pytest.mark.parametrize(
    "recipe_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {
                "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
                "override_folder_name": "test_folder",
            },
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned",
                "optim_category": "conservative",
                "optim_target": None,
            },
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet_v1",
                "sub_architecture": "50",
                "framework": "pytorch",
                "repo": "sparseml",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "pruned_quant",
                "optim_category": "moderate",
                "optim_target": None,
                "recipe_type": "transfer_learn",
            },
            {},
        ),
    ],
)
def test_load_base_model_from_recipe(recipe_args, other_args):
    recipe = Zoo.load_recipe(**recipe_args, **other_args)
    recipe_model = Zoo.load_base_model_from_recipe(recipe, **other_args)
    model_dict = recipe_model.dict()
    for field, value in recipe_args.items():
        if field == "recipe_type":
            continue
        if field == "optim_name":
            assert model_dict[field] == "base"
        elif field == "optim_category":
            assert model_dict[field] == "none"
        elif field == "optim_target":
            assert model_dict[field] is None
        else:
            assert model_dict[field] == value
