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
Code related to a model repo recipe file
"""

import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union


if TYPE_CHECKING:
    from sparsezoo.objects.model import Model

from enum import Enum

from sparsezoo.objects.file import File
from sparsezoo.objects.metadata import ModelMetadata
from sparsezoo.requests import (
    ModelArgs,
    RecipeArgs,
    download_recipe_get_request,
    get_recipe_get_request,
    parse_zoo_stub,
    search_recipe_get_request,
)


__all__ = ["RecipeTypes", "Recipe"]


_LOGGER = logging.getLogger(__name__)


class RecipeTypes(Enum):
    """
    Types of recipes available in the sparsezoo
    """

    ORIGINAL = "original"
    TRANSFER_LEARN = "transfer_learn"


class Recipe(File):
    """
    A model repo recipe.

    :param model_metadata: the metadata for the model the file is for
    :param recipe_id: the recipe id
    :param recipe_type: the type of recipe
    :param display_description: the display description for the recipe
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
        super(Recipe, self).__init__(
            model_metadata=model_metadata, child_folder_name="recipes", **kwargs
        )
        self._recipe_id = recipe_id
        self._recipe_type = recipe_type
        self._display_description = display_description
        self._base_stub = base_stub
        self._model = None
        self._base_model = None

    @staticmethod
    def load_recipe(
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: Union[str, None],
        framework: str,
        repo: str,
        dataset: str,
        training_scheme: Union[str, None],
        sparse_name: str,
        sparse_category: str,
        sparse_target: Union[str, None],
        recipe_type: Union[str, None] = None,
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Recipe":
        """
        Obtains a Recipe from the model repo

        :param domain: The domain of the model the object belongs to;
            e.g. cv, nlp
        :param sub_domain: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        :param architecture: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        :param sub_architecture: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        :param framework: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param repo: The source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param dataset: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param training_scheme: The training scheme used on the model the object
            belongs to if any; e.g. augmented
        :param sparse_name: The name describing the sparsification of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param sparse_category: The degree of sparsification of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param sparse_target: The deployment target of sparsification of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :param recipe_type: The recipe type; e.g. original, transfer_learn
        :param release_version: The sparsezoo release version for the model
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Recipe instance
        """
        args = ModelArgs(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            sparse_name=sparse_name,
            sparse_category=sparse_category,
            sparse_target=sparse_target,
            release_version=release_version,
        )
        _LOGGER.debug("load_recipe: loading recipe")
        return Recipe.load_recipe_from_stub(
            args,
            recipe_type=recipe_type,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def load_recipe_from_stub(
        stub: Union[str, ModelArgs],
        recipe_type: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Recipe":
        """
        Loads a recipe from stub. If the stub is a string, it may contain the
        recipe type as a stub parameter. i.e.
            - "model/stub/path"
            - "zoo:model/stub/path",
            - "zoo:model/stub/path?recipe_type=original",
            - "zoo:model/stub/path/transfer_learn"

        :param stub: the SparseZoo stub path to the recipe, can be a string path or
            ModelArgs object
        :param recipe_type: the recipe type to obtain if not original
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Recipe instance
        """
        if isinstance(stub, str):
            stub, args = parse_zoo_stub(stub, valid_params=["recipe_type"])
            if recipe_type is None:
                recipe_type = _get_stub_args_recipe_type(args)

        _LOGGER.debug("load_recipe_from_stub: loading " + f"recipe from stub {stub}")
        response_json = get_recipe_get_request(
            args=stub,
            recipe_type=recipe_type,
            force_token_refresh=force_token_refresh,
        )

        recipe = response_json["recipe"]
        return Recipe(
            **recipe,
            model_metadata=recipe["model"],
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
        )

    @staticmethod
    def download_recipe(
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: Union[str, None],
        framework: str,
        repo: str,
        dataset: str,
        training_scheme: Union[str, None],
        sparse_name: str,
        sparse_category: str,
        sparse_target: Union[str, None],
        recipe_type: Union[str, None] = None,
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Recipe":
        """
        Downloads a Recipe from the model repo

        :param domain: The domain of the model the object belongs to;
            e.g. cv, nlp
        :param sub_domain: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        :param architecture: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        :param sub_architecture: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        :param framework: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param repo: The source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param dataset: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param training_scheme: The training scheme used on the model the object
            belongs to if any; e.g. augmented
        :param sparse_name: The name describing the sparsification of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param sparse_category: The degree of sparsification of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param sparse_target: The deployment target of sparsification of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :param recipe_type: The recipe type; e.g. original, transfer_learn
        :param release_version: The sparsezoo release version for the model
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Recipe instance
        """
        args = ModelArgs(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            sparse_name=sparse_name,
            sparse_category=sparse_category,
            sparse_target=sparse_target,
            release_version=release_version,
        )
        _LOGGER.debug("download_recipe: downloading recipe")
        return Recipe.download_recipe_from_stub(
            args,
            recipe_type=recipe_type,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def download_recipe_from_stub(
        stub: Union[str, ModelArgs],
        recipe_type: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
        overwrite: bool = False,
    ) -> "Recipe":
        """
        Downloads a recipe from stub. If the stub is a string, it may contain the
        recipe type as a stub parameter or part of the stub. i.e.
            - "model/stub/path"
            - "zoo:model/stub/path",
            - "zoo:model/stub/path?recipe_type=original",
            - "zoo:model/stub/path/transfer_learn"

        :param stub: the SparseZoo stub path to the recipe, can be a string path or
            ModelArgs object
        :param recipe_type: the recipe_type to download if not original
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :param overwrite: True to overwrite the file if it exists, False otherwise
        :return: The requested Recipe instance
        """
        if isinstance(stub, str):
            stub, args = parse_zoo_stub(stub, valid_params=["recipe_type"])
            if recipe_type is None:
                recipe_type = _get_stub_args_recipe_type(args)
        _LOGGER.debug(
            "download_recipe_from_stub: downloading " + f"recipe from stub {stub}"
        )
        response_json = download_recipe_get_request(stub, recipe_type=recipe_type)

        recipe = response_json["recipe"]

        recipe = Recipe(
            **recipe,
            model_metadata=recipe["model"],
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
        )
        recipe.download(overwrite=overwrite)
        return recipe

    @staticmethod
    def search_recipes(
        domain: str,
        sub_domain: str,
        architecture: Union[str, None] = None,
        sub_architecture: Union[str, None] = None,
        framework: Union[str, None] = None,
        repo: Union[str, None] = None,
        dataset: Union[str, None] = None,
        training_scheme: Union[str, None] = None,
        sparse_name: Union[str, None] = None,
        sparse_category: Union[str, None] = None,
        sparse_target: Union[str, None] = None,
        release_version: Union[str, None] = None,
        recipe_type: Union[str, None] = None,
        page: int = 1,
        page_length: int = 20,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> List["Recipe"]:
        """
        Obtains a list of Recipes matching the model search parameters

        :param domain: The domain of the model the object belongs to;
            e.g. cv, nlp
        :param sub_domain: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        :param architecture: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        :param sub_architecture: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        :param framework: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param repo: The source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param dataset: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param training_scheme: The training scheme used on the model the object
            belongs to if any; e.g. augmented
        :param sparse_name: The name describing the sparsification of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param sparse_category: The degree of sparsification of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param sparse_target: The deployment target of sparsification of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :param release_version: The sparsezoo release version for the model
        :param recipe_type: The recipe type; e.g. original, transfer_learn
        :param page: the page of values to get
        :param page_length: the page length of values to get
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: A list of Recipe objects for models that match the given
            search parameters
        """
        args = RecipeArgs(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            sparse_name=sparse_name,
            sparse_category=sparse_category,
            sparse_target=sparse_target,
            release_version=release_version,
            recipe_type=recipe_type,
        )
        _LOGGER.debug(
            ("search_recipes: searching recipes " + f"with args {args.model_url_args}")
        )
        response_json = search_recipe_get_request(
            args=args,
            page=page,
            page_length=page_length,
            force_token_refresh=force_token_refresh,
        )

        return [
            Recipe(
                **recipe,
                model_metadata=recipe["model"],
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            for recipe in response_json["recipes"]
        ]

    @staticmethod
    def search_sparse_recipes(
        model: Union["Model", str, ModelArgs],
        recipe_type: Union[str, None] = None,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List["Recipe"]:
        """
        Search for recipes of the given model

        :param model: The model object, a SparseZoo stub model path, or a ModelArgs
            object representing the base model to search for recipes
        :param match_framework: True to match similar models to the current
            framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param match_repo: True to match similar models to the current
            source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param match_dataset: True to match similar models to the current
            dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param match_training_scheme: True to match similar models to the current
            training scheme used on the model the object
            belongs to if any; e.g. augmented
        :return: the list of matching recipes, if any
        """
        from sparsezoo.objects.model import Model

        if isinstance(recipe_type, str):
            recipe_type = RecipeTypes(recipe_type).value

        if not isinstance(model, Model):
            model = Model.load_model_from_stub(model)

        _LOGGER.debug(
            ("search_sparse_recipes: searching for recipes " + f"from model {model}")
        )

        sparse_models = model.search_similar_models(
            match_domain=True,
            match_sub_domain=True,
            match_architecture=True,
            match_sub_architecture=True,
            match_framework=match_framework,
            match_repo=match_repo,
            match_dataset=match_dataset,
            match_training_scheme=match_training_scheme,
        )
        return [
            recipe
            for model in sparse_models
            for recipe in model.recipes
            if recipe_type is None or recipe_type == recipe.recipe_type
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}(stub={self.stub})"

    def __str__(self):
        return f"{self.__class__.__name__}(stub={self.stub})"

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
        return self.recipe_type == RecipeTypes.ORIGINAL.value

    @property
    def recipe_type_transfer_learn(self) -> bool:
        """
        :return: True if this is a recipe for transfer learning from the
            created model, False otherwise
        """
        return self.recipe_type == RecipeTypes.TRANSFER_LEARN.value

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

    def load_model(
        self,
        override_folder_name: Optional[str] = None,
        override_parent_path: Optional[str] = None,
    ) -> "Model":
        """
        :return: the model associated with the recipe
        """
        if self._model is None:
            from sparsezoo.objects.model import Model

            self._model = Model.load_model_from_recipe(
                recipe=self,
                override_folder_name=os.path.dirname(self.folder_name),
                override_parent_path=self.override_parent_path,
            )

        return self._model

    def load_base_model(
        self,
        override_folder_name: Optional[str] = None,
        override_parent_path: Optional[str] = None,
    ) -> Optional["Model"]:
        """
        :return: the base model associated with the recipe
        """
        if self._base_model is None:
            from sparsezoo.objects.model import Model

            self._base_model = Model.load_base_model_from_recipe(
                recipe=self,
                override_folder_name=os.path.dirname(self.folder_name),
                override_parent_path=self.override_parent_path,
            )
        return self._base_model

    def download_base_framework_files(
        self,
        override_folder_name: Optional[str] = None,
        override_parent_path: Optional[str] = None,
        force_token_refresh: bool = False,
        overwrite: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> List[str]:
        """
        :param stub: a string model stub that points to a SparseZoo model.
            recipe_type may be added as a stub parameter or path of path. i.e.
            "model/stub/path", "zoo:model/stub/path",
            "zoo:model/stub/path?recipe_type=transfer",
            "zoo:model/stub/path/transfer"
        :param recipe_type: the recipe_type to download if not original
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param extensions: List of file extensions to filter for. ex ['.pth', '.ptc'].
            If None or empty list, all framework files are downloaded. Default is None
        :return: file path to the downloaded framework checkpoint files for the
            base weights of this recipe
        """
        _LOGGER.debug(
            "download_base_framework_files: download"
            + f" base framework files from recipe {self.stub}"
        )

        if self.recipe_type == RecipeTypes.TRANSFER_LEARN.value:
            model = self.load_model(
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            # return final model's sparse weights for sparse transfer learning
            framework_files = model.download_framework_files(extensions=extensions)

            # download only pre-quantized weights if available
            checkpoint_framework_files = [
                framework_file
                for framework_file in framework_files
                if ".ckpt" in framework_file
            ]

            # return non-empty list, preferring filtered list
            return checkpoint_framework_files or framework_files
        else:
            # search for base model, and return those weights as a starting checkpoint
            if not self.base_stub:
                raise ValueError(f"Could not find base model for recipe {self.stub}")

            base_model = self.load_base_model(
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            framework_files = base_model.download_framework_files(extensions=extensions)

            # filter out checkpoint weights if any exist
            base_framework_files = [
                framework_file
                for framework_file in framework_files
                if ".ckpt" not in framework_file
            ]

            # return non-empty list, preferring filtered list
            return base_framework_files or framework_files


def _get_stub_args_recipe_type(stub_args: Dict[str, str]) -> str:
    # check recipe type, default to original, and validate
    recipe_type = stub_args.get("recipe_type")

    # validate
    valid_recipe_types = list(map(lambda typ: typ.value, RecipeTypes))
    if recipe_type not in valid_recipe_types and recipe_type is not None:
        raise ValueError(
            f"Invalid recipe_type: '{recipe_type}'. "
            f"Valid recipe types: {valid_recipe_types}"
        )
    return recipe_type
