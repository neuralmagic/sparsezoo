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
from typing import Dict, List, Tuple, Union

from sparsezoo.objects.model import Model
from sparsezoo.objects.optimization_recipe import (
    OptimizationRecipe,
    OptimizationRecipeTypes,
)
from sparsezoo.requests import ModelArgs, download_get_request, search_get_request


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
    ) -> Model:
        """
        Obtains a Model with signed files from the model repo

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
        if isinstance(stub, str):
            stub, _ = parse_zoo_stub(stub, valid_params=[])

        response_json = download_get_request(
            args=stub,
            file_name=None,
            force_token_refresh=force_token_refresh,
        )
        return Model(
            **response_json["model"],
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
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
        optim_name: Union[str, None] = None,
        optim_category: Union[str, None] = None,
        optim_target: Union[str, None] = None,
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
        response_json = search_get_request(
            args=args,
            page=page,
            page_length=page_length,
            force_token_refresh=force_token_refresh,
        )

        return [
            Model(
                **model,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            for model in response_json["models"]
        ]

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
        match_optim_name: bool = False,
        match_optim_category: bool = False,
        match_optim_target: bool = False,
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
        if isinstance(model, Model) or isinstance(model, str):
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
        model: Union[Model, str, ModelArgs],
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List[Model]:
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
        return Zoo.search_similar_models(
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
        page: int = 1,
        page_length: int = 20,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> List[OptimizationRecipe]:
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
        :param page: the page of values to get
        :param page_length: the page length of values to get
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: A list of OptimizationRecipe objects for models that match the given
            search parameters
        """
        matched_models = Zoo.search_models(
            domain,
            sub_domain,
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
            page=page,
            page_length=page_length,
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
            force_token_refresh=force_token_refresh,
        )
        return [recipe for model in matched_models for recipe in model.recipes]

    @staticmethod
    def search_optimized_recipes(
        model: Union[Model, str, ModelArgs],
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List[OptimizationRecipe]:
        """
        Search for optimized recipes of the given model

        :param model: The model object, a SparseZoo stub model path, or a ModelArgs
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
        :return: the list of matching optimization recipes, if any
        """
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
        return [recipe for model in optimized_models for recipe in model.recipes]

    @staticmethod
    def download_recipe_from_stub(
        stub: str,
    ) -> str:
        """
        :param stub: a string model stub that points to a SparseZoo model.
            recipe_type may be added as a stub parameter. i.e.
            "model/stub/path", "zoo:model/stub/path",
            "zoo:model/stub/path?recipe_type=original"
        :return: file path of the downloaded recipe for that model
        """
        stub, args = parse_zoo_stub(stub, valid_params=["recipe_type"])
        recipe_type = _get_stub_args_recipe_type(args)
        model = Zoo.load_model_from_stub(stub)

        for recipe in model.recipes:
            if recipe.recipe_type == recipe_type:
                return recipe.downloaded_path()

        found_recipe_types = [recipe.recipe_type for recipe in model.recipes]
        raise RuntimeError(
            f"No recipe with recipe_type {recipe_type} found for model {model}. "
            f"Found {len(model.recipes)} recipes with recipe types {found_recipe_types}"
        )

    @staticmethod
    def download_recipe_base_framework_files(
        stub: str,
        extensions: Union[List[str], None] = None,
    ) -> List[str]:
        """
        :param stub: a string model stub that points to a SparseZoo model.
            recipe_type may be added as a stub parameter. i.e.
            "model/stub/path", "zoo:model/stub/path",
            "zoo:model/stub/path?recipe_type=transfer"
        :param extensions: List of file extensions to filter for. ex ['.pth', '.ptc'].
            If None or empty list, all framework files are downloaded. Default is None
        :return: file path to the downloaded framework checkpoint files for the
            base weights of this recipe
        """
        stub, args = parse_zoo_stub(stub, valid_params=["recipe_type"])
        recipe_type = _get_stub_args_recipe_type(args)
        model = Zoo.load_model_from_stub(stub)

        if recipe_type == OptimizationRecipeTypes.TRANSFER_LEARN.value:
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
            base_model = [
                result
                for result in Zoo.search_optimized_models(model)
                if result.optim_name == "base"
            ]
            if not base_model:
                raise ValueError(f"Could not find base model for model {model}")
            framework_files = base_model[0].download_framework_files(
                extensions=extensions
            )

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
    recipe_type = stub_args.get("recipe_type", OptimizationRecipeTypes.ORIGINAL.value)

    # validate
    valid_recipe_types = list(map(lambda typ: typ.value, OptimizationRecipeTypes))
    if recipe_type not in valid_recipe_types:
        raise ValueError(
            f"Invalid recipe_type: '{recipe_type}'. "
            f"Valid recipe types: {valid_recipe_types}"
        )
    return recipe_type
