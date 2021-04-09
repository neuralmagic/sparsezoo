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
Code for managing the search and creation of sparsezoo Model and Recipe objects
"""


from typing import List, Union

from sparsezoo.objects.model import Model
from sparsezoo.objects.recipe import Recipe
from sparsezoo.requests import ModelArgs


__all__ = [
    "Zoo",
]


class Zoo:
    """
    Provides static functions for loading and searching SparseZoo models and recipes
    """

    @staticmethod
    def load_model(
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
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> Model:
        """
        Obtains a Model from the model repo

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
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
        """
        return Model.load_model(
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
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def load_model_from_stub(
        stub: Union[str, ModelArgs],
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> Model:
        """
        :param stub: the SparseZoo stub path to the model, can be a string path or
            ModelArgs object
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
        """
        return Model.load_model_from_stub(
            stub=stub,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def load_model_from_recipe(
        recipe: Recipe,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ):
        """
        Loads the model associated with a recipe

        :param recipe: the Recipe associated with the model
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
        """
        return Model.load_model_from_recipe(
            recipe=recipe,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def load_base_model_from_recipe(
        recipe: Recipe,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ):
        """
        Loads the base model associated with a recipe

        :param recipe: the Recipe associated with the model
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
        """
        return Model.load_base_model_from_recipe(
            recipe=recipe,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def download_model(
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
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
        overwrite: bool = False,
    ) -> Model:
        """
        Downloads a model from model repo

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
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :param overwrite: True to overwrite the file if it exists, False otherwise
        :return: The requested Model instance
        """
        return Model.download_model(
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
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
            overwrite=overwrite,
        )

    @staticmethod
    def download_model_from_stub(
        stub: Union[str, ModelArgs],
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
        overwrite: bool = False,
    ) -> Model:
        """
        :param stub: the SparseZoo stub path to the model, can be a string path or
            ModelArgs object
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :param overwrite: True to overwrite the file if it exists, False otherwise
        :return: The requested Model instance
        """
        return Model.download_model_from_stub(
            stub=stub,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
            overwrite=overwrite,
        )

    @staticmethod
    def search_models(
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
        page: int = 1,
        page_length: int = 20,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> List[Model]:
        """
        Obtains a list of Models matching the search parameters

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
        :param page: the page of values to get
        :param page_length: the page length of values to get
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
        """
        return Model.search_models(
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
            page=page,
            page_length=page_length,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def search_similar_models(
        model: Union[Model, str, ModelArgs],
        match_domain: bool = True,
        match_sub_domain: bool = True,
        match_architecture: bool = True,
        match_sub_architecture: bool = True,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = False,
        match_sparse_name: bool = False,
        match_sparse_category: bool = False,
        match_sparse_target: bool = False,
    ) -> List[Model]:
        """
        Search for similar models to the given one

        :param model: The model object, a SparseZoo model stub path, or a ModelArgs
            object representing the base model to search similar models of
        :param match_domain: True to match similar models to the current
            domain of the model the object belongs to; e.g. cv, nlp
        :param match_sub_domain: True to match similar models to the current
            sub domain of the model the object belongs to;
            e.g. classification, segmentation
        :param match_architecture: True to match similar models to the current
            architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        :param match_sub_architecture: True to match similar models to the current
            sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
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
        :param match_sparse_name: True to match similar models to the current
            name describing the sparsification of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param match_sparse_category: True to match similar models to the current
            degree of sparsification of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param match_sparse_target: True to match similar models to the current
            deployment target of sparsification of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :return: a list of models matching the given model, if any
        """
        if isinstance(model, str):
            model = Zoo.load_model_from_stub(model)
        return model.search_similar_models(
            match_domain=match_domain,
            match_sub_domain=match_sub_domain,
            match_architecture=match_architecture,
            match_sub_architecture=match_sub_architecture,
            match_framework=match_framework,
            match_repo=match_repo,
            match_dataset=match_dataset,
            match_training_scheme=match_training_scheme,
            match_sparse_name=match_sparse_name,
            match_sparse_category=match_sparse_category,
            match_sparse_target=match_sparse_target,
        )

    @staticmethod
    def search_sparse_models(
        model: Union[Model, str, ModelArgs],
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List[Model]:
        """
        Search for different available sparse versions based off of the current model

        :param model: The model object, a SparseZoo model stub path, or a ModelArgs
            object representing the base model to search different sparsifications of
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
        :return: the list of matching sparse models, if any
        """
        if isinstance(model, str):
            model = Zoo.load_model_from_stub(model)
        return model.search_sparse_models(
            match_framework=match_framework,
            match_repo=match_repo,
            match_dataset=match_dataset,
            match_training_scheme=match_training_scheme,
        )

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
    ) -> List[Recipe]:
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
        return Recipe.search_recipes(
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
            page=page,
            page_length=page_length,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def search_sparse_recipes(
        model: Union[Model, str, ModelArgs],
        recipe_type: Union[str, None] = None,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List[Recipe]:
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
        :return: the list of matching sparsification recipes, if any
        """
        return Recipe.search_sparse_recipes(
            model=model,
            recipe_type=recipe_type,
            match_framework=match_framework,
            match_repo=match_repo,
            match_dataset=match_dataset,
            match_training_scheme=match_training_scheme,
        )

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
    ) -> Recipe:
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
        return Recipe.load_recipe(
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
    ) -> Recipe:
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
        return Recipe.load_recipe_from_stub(
            stub=stub,
            recipe_type=recipe_type,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
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
    ) -> Model:
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
        return Recipe.download_recipe(
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
    ) -> Recipe:
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
        return Recipe.download_recipe_from_stub(
            stub=stub,
            recipe_type=recipe_type,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
            overwrite=overwrite,
        )

    @staticmethod
    def download_recipe_base_framework_files(
        stub: Union[str, ModelArgs],
        recipe_type: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
        overwrite: bool = False,
        extensions: Union[List[str], None] = None,
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
        recipe = Zoo.load_recipe_from_stub(
            stub,
            recipe_type=recipe_type,
            force_token_refresh=force_token_refresh,
        )
        return recipe.download_base_framework_files(
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            overwrite=overwrite,
            extensions=extensions,
        )
