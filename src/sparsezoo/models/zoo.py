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


import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union


if TYPE_CHECKING:
    from sparsezoo.objects.model import Model
    from sparsezoo.objects.recipe import OptimizationRecipe

from sparsezoo.requests import (
    ModelArgs,
    RecipeArgs,
    download_model_get_request,
    download_recipe_get_request,
    get_model_get_request,
    get_recipe_get_request,
    search_model_get_request,
    search_recipe_get_request,
)


__all__ = [
    "ZOO_STUB_PREFIX",
    "parse_zoo_stub",
    "Zoo",
]


# optional prefix for stubs
ZOO_STUB_PREFIX = "zoo:"


def parse_zoo_stub(
    stub: str, valid_params: Union[List[str], None] = None
) -> Tuple[str, Dict[str, str]]:
    """
    :param stub: A SparseZoo model stub. i.e. 'model/stub/path',
        'zoo:model/stub/path', 'zoo:model/stub/path?param1=value1&param2=value2'
    :param valid_params: list of expected parameter names to be encoded in the
        stub. Will raise a warning if any unexpected param names are given. Leave
        as None to not raise any warnings. Default is None
    :return: the parsed base stub and a dictionary of parameter names and their values
    """
    # strip optional zoo stub prefix
    if stub.startswith(ZOO_STUB_PREFIX):
        stub = stub[len(ZOO_STUB_PREFIX) :]

    if "?" not in stub:
        return stub, {}

    stub_parts = stub.split("?")
    if len(stub_parts) > 2:
        raise ValueError(
            "Invalid SparseZoo stub, query string must be preceded by only one '?'"
            f"given {stub}"
        )
    stub, params = stub_parts
    params = dict(param.split("=") for param in params.split("&"))

    if valid_params is not None and any(param not in valid_params for param in params):
        warnings.warn(
            f"Invalid query string for stub {stub} valid params include {valid_params},"
            f" given {list(params.keys())}"
        )

    return stub, params


class Zoo:
    """
    Provides static functions for loading and searching SparseZoo models and recipes
    """

    _CONSTRUCTORS = {}

    @staticmethod
    def constructor(construct_key: str):
        """
        :param construct_key: The constructor key
        :return: Constructor registered with the key
        """
        if construct_key not in Zoo._CONSTRUCTORS:
            raise RuntimeError(f"No constructure registered for {construct_key}")
        return Zoo._CONSTRUCTORS[construct_key]

    @staticmethod
    def _register_constructor(key: str, const_func: Callable):
        Zoo._CONSTRUCTORS[key] = const_func

    @staticmethod
    def register(
        key: str,
    ):
        """
        Registers a constructor with key

        :param key: The key the constructor will be registered with
        """

        def decorator(const_func: Callable):
            Zoo._register_constructor(key, const_func)

        return decorator

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
        optim_name: str,
        optim_category: str,
        optim_target: Union[str, None],
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Model":
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
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :param release_version: The sparsezoo release version for the model
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
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
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
        )
        return Zoo.load_model_from_stub(
            args, override_folder_name, override_parent_path, force_token_refresh
        )

    @staticmethod
    def load_model_from_stub(
        stub: Union[str, ModelArgs],
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Model":
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
        if isinstance(stub, str):
            stub, _ = parse_zoo_stub(stub, valid_params=[])

        response_json = get_model_get_request(
            args=stub,
            file_name=None,
            force_token_refresh=force_token_refresh,
        )
        return Zoo.constructor("Model")(
            **response_json["model"],
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
        )

    @staticmethod
    def load_model_from_recipe(
        recipe: "OptimizationRecipe",
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
        return Zoo.load_model_from_stub(
            stub=recipe.model_metadata,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )

    @staticmethod
    def load_base_model_from_recipe(
        recipe: "OptimizationRecipe",
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
        return Zoo.load_model_from_stub(
            recipe.base_stub,
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
        optim_name: str,
        optim_category: str,
        optim_target: Union[str, None],
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
        overwrite: bool = False,
    ) -> "Model":
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
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
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
        args = ModelArgs(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
        )
        return Zoo.download_model_from_stub(
            args,
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
    ) -> "Model":
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
        if isinstance(stub, str):
            stub, _ = parse_zoo_stub(stub, valid_params=[])

        response_json = download_model_get_request(
            args=stub,
            file_name=None,
            force_token_refresh=force_token_refresh,
        )
        model = Zoo.constructor("Model")(
            **response_json["model"],
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
        )
        model.download(overwrite=overwrite, refresh_token=force_token_refresh)
        return model

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
        optim_name: Union[str, None] = None,
        optim_category: Union[str, None] = None,
        optim_target: Union[str, None] = None,
        release_version: Union[str, None] = None,
        page: int = 1,
        page_length: int = 20,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> List["Model"]:
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
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
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
        args = ModelArgs(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
        )
        response_json = search_model_get_request(
            args=args,
            page=page,
            page_length=page_length,
            force_token_refresh=force_token_refresh,
        )

        return [
            Zoo.constructor("Model")(
                **model,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            for model in response_json["models"]
        ]

    @staticmethod
    def search_similar_models(
        model: Union["Model", str, ModelArgs],
        match_domain: bool = True,
        match_sub_domain: bool = True,
        match_architecture: bool = True,
        match_sub_architecture: bool = True,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = False,
        match_optim_name: bool = False,
        match_optim_category: bool = False,
        match_optim_target: bool = False,
    ) -> List["Model"]:
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
        :param match_optim_name: True to match similar models to the current
            name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param match_optim_category: True to match similar models to the current
            degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param match_optim_target: True to match similar models to the current
            deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :return: a list of models matching the given model, if any
        """
        if isinstance(model, str):
            model = Zoo.load_model_from_stub(model)
        return Zoo.search_models(
            domain=model.domain if match_domain else None,
            sub_domain=model.sub_domain if match_sub_domain else None,
            architecture=model.architecture if match_architecture else None,
            sub_architecture=model.sub_architecture if match_sub_architecture else None,
            framework=model.framework if match_framework else None,
            repo=model.repo if match_repo else None,
            dataset=model.dataset if match_dataset else None,
            training_scheme=model.training_scheme if match_training_scheme else None,
            optim_name=model.optim_name if match_optim_name else None,
            optim_category=model.optim_category if match_optim_category else None,
            optim_target=model.optim_target if match_optim_target else None,
        )

    @staticmethod
    def search_optimized_models(
        model: Union["Model", str, ModelArgs],
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List["Model"]:
        """
        Search for different available optimized versions based off of the current model

        :param model: The model object, a SparseZoo model stub path, or a ModelArgs
            object representing the base model to search different optimizations of
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
        :return: the list of matching optimized models, if any
        """
        return [
            model
            for model in Zoo.search_similar_models(
                model=model,
                match_domain=True,
                match_sub_domain=True,
                match_architecture=True,
                match_sub_architecture=True,
                match_framework=match_framework,
                match_repo=match_repo,
                match_dataset=match_dataset,
                match_training_scheme=match_training_scheme,
            )
            if not model.is_base
        ]

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
        optim_name: Union[str, None] = None,
        optim_category: Union[str, None] = None,
        optim_target: Union[str, None] = None,
        release_version: Union[str, None] = None,
        recipe_type: Union[str, None] = None,
        page: int = 1,
        page_length: int = 20,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> List["OptimizationRecipe"]:
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
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
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
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
            recipe_type=recipe_type,
        )
        response_json = search_recipe_get_request(
            args=args,
            page=page,
            page_length=page_length,
            force_token_refresh=force_token_refresh,
        )

        return [
            Zoo.constructor("OptimizationRecipe")(
                **recipe,
                model_metadata=recipe["model"],
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            for recipe in response_json["recipes"]
        ]

    @staticmethod
    def search_optimized_recipes(
        model: Union["Model", str, ModelArgs],
        recipe_type: Union[str, None] = None,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List["OptimizationRecipe"]:
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
        :return: the list of matching optimization recipes, if any
        """
        if isinstance(recipe_type, str):
            recipe_type = Zoo.constructor("OptimizationRecipeTypes")(recipe_type).value

        optimized_models = Zoo.search_similar_models(
            model=model,
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
            for model in optimized_models
            for recipe in model.recipes
            if recipe_type is None or recipe_type == recipe.recipe_type
        ]

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
        optim_name: str,
        optim_category: str,
        optim_target: Union[str, None],
        recipe_type: Union[str, None] = None,
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Model":
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
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
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
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
        )
        return Zoo.load_recipe_from_stub(
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
    ) -> "OptimizationRecipe":
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

        response_json = get_recipe_get_request(
            args=stub,
            recipe_type=recipe_type,
            force_token_refresh=force_token_refresh,
        )

        recipe = response_json["recipe"]
        return Zoo.constructor("OptimizationRecipe")(
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
        optim_name: str,
        optim_category: str,
        optim_target: Union[str, None],
        recipe_type: Union[str, None] = None,
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Model":
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
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
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
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
        )
        return Zoo.download_recipe_from_stub(
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
    ) -> "OptimizationRecipe":
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

        response_json = download_recipe_get_request(stub, recipe_type=recipe_type)

        recipe = response_json["recipe"]

        recipe = Zoo.constructor("OptimizationRecipe")(
            **recipe,
            model_metadata=recipe["model"],
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
        )
        recipe.download(overwrite=overwrite)
        return recipe

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

        OptimizationRecipeTypes = Zoo.constructor("OptimizationRecipeTypes")

        if recipe.recipe_type == OptimizationRecipeTypes.TRANSFER_LEARN.value:
            model = Zoo.load_model_from_recipe(
                recipe,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            # return final model's optimized weights for sparse transfer learning
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
            if not recipe.base_stub:
                raise ValueError(f"Could not find base model for recipe {recipe.stub}")

            base_model = Zoo.load_base_model_from_recipe(
                recipe,
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
    OptimizationRecipeTypes = Zoo.constructor("OptimizationRecipeTypes")

    valid_recipe_types = list(map(lambda typ: typ.value, OptimizationRecipeTypes))
    if recipe_type not in valid_recipe_types and recipe_type is not None:
        raise ValueError(
            f"Invalid recipe_type: '{recipe_type}'. "
            f"Valid recipe types: {valid_recipe_types}"
        )
    return recipe_type
