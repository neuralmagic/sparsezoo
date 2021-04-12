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
Code related to a model from the sparsezoo
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import numpy

from sparsezoo.objects.data import Data
from sparsezoo.objects.downloadable import Downloadable
from sparsezoo.objects.file import File, FileTypes
from sparsezoo.objects.metadata import ModelMetadata
from sparsezoo.objects.recipe import Recipe
from sparsezoo.objects.release_version import ReleaseVersion
from sparsezoo.objects.result import Result
from sparsezoo.objects.tag import Tag
from sparsezoo.objects.user import User
from sparsezoo.requests import (
    ModelArgs,
    download_model_get_request,
    get_model_get_request,
    parse_zoo_stub,
    search_model_get_request,
)
from sparsezoo.utils import DataLoader


__all__ = ["Model"]

_LOGGER = logging.getLogger(__name__)


class Model(Downloadable, ModelMetadata):
    """
    A model repo model

    :param display_name: the display name for the model
    :param display_description: the description for the model
    :param files: a list of model repo files for this model
    :param recipes: a list of model repo recipes for this model
    :param results: a list of model repo results for this model
    :param tags: a list of model repo tags for this model
    :param user: the model repo user who uploaded this model
    :param release_version: a model repo release version this model was released with
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
    """

    def __init__(
        self,
        display_name: str,
        display_description: str,
        files: List[Dict[str, Any]] = None,
        recipes: List[Dict[str, Any]] = None,
        results: List[Dict[str, Any]] = None,
        tags: List[Dict[str, Any]] = None,
        user: Dict[str, Any] = None,
        release_version: Dict[str, Any] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        **kwargs,
    ):
        release_version = ReleaseVersion(**release_version) if release_version else None

        super(Model, self).__init__(
            **kwargs,
            folder_name=kwargs["model_id"]
            if not override_folder_name
            else override_folder_name,
            override_parent_path=override_parent_path,
        )
        super(Downloadable, self).__init__(**kwargs, release_version=release_version)

        self._results = [Result(**res) for res in results] if results else []
        self._tags = [Tag(**res) for res in tags] if tags else []
        self._user = User(**user) if user else None
        self._release_version = release_version

        metadata = ModelMetadata(**kwargs, release_version=self._release_version)
        sorted_files = {
            file_type.value: [
                res for res in files if res["file_type"] == file_type.value
            ]
            for file_type in FileTypes
        }

        self._display_name = display_name
        self._display_description = display_description

        self._recipes = (
            [
                Recipe(
                    model_metadata=metadata,
                    override_folder_name=override_folder_name,
                    override_parent_path=override_parent_path,
                    **res,
                )
                for res in recipes
            ]
            if recipes
            else []
        )

        self._data = OrderedDict()
        if sorted_files[FileTypes.DATA_INPUTS.value]:
            self._data["inputs"] = Data(
                name="inputs",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_INPUTS.value][0],
            )
        if sorted_files[FileTypes.DATA_OUTPUTS.value]:
            self._data["outputs"] = Data(
                name="outputs",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_OUTPUTS.value][0],
            )
        if sorted_files[FileTypes.DATA_LABELS.value]:
            self._data["labels"] = Data(
                name="labels",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_LABELS.value][0],
            )
        if sorted_files[FileTypes.DATA_ORIGINALS.value]:
            self._data["originals"] = Data(
                name="originals",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_ORIGINALS.value][0],
            )

        self._framework_files = [
            File(
                model_metadata=metadata,
                child_folder_name=metadata.framework,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **res,
            )
            for res in sorted_files[FileTypes.FRAMEWORK.value]
        ]

        self._onnx_files = (
            [
                File(
                    model_metadata=metadata,
                    override_folder_name=override_folder_name,
                    override_parent_path=override_parent_path,
                    **res,
                )
                for res in sorted_files[FileTypes.ONNX.value]
            ]
            if sorted_files[FileTypes.ONNX.value]
            else None
        )
        if sorted_files[FileTypes.ONNX_GZ.value]:
            if not self._onnx_files:
                self._onnx_files = []
            self._onnx_files.extend(
                [
                    File(
                        model_metadata=metadata,
                        override_folder_name=override_folder_name,
                        override_parent_path=override_parent_path,
                        **res,
                    )
                    for res in sorted_files[FileTypes.ONNX_GZ.value]
                ]
            )

        self._card_file = (
            File(
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.CARD.value][0],
            )
            if sorted_files[FileTypes.CARD.value]
            else None
        )

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
        _LOGGER.debug("load_model: loading model")
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
        return Model.load_model_from_stub(
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
        _LOGGER.debug(f"load_model_from_stub: loading model from {stub}")
        response_json = get_model_get_request(
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
    def load_model_from_recipe(
        recipe: Recipe,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> "Model":
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
        _LOGGER.debug(f"load_model_from_recipe: loading model from recipe {recipe}")
        return Model.load_model_from_stub(
            stub=recipe.model_metadata,
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
    ) -> Optional["Model"]:
        """
        Loads the base model associated with a recipe if any exists

        :param recipe: the Recipe associated with the model
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
        """
        if recipe.base_stub is None:
            _LOGGER.warn(
                "load_base_model_from_recipe: No base model found for "
                + f"recipe {recipe}"
            )
            return None
        _LOGGER.debug(
            (
                f"load_base_model_from_recipe: loading base model with {recipe}"
                + f" with stub {recipe.base_stub}"
            )
        )
        return Model.load_model_from_stub(
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
        sparse_name: str,
        sparse_category: str,
        sparse_target: Union[str, None],
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
        _LOGGER.debug("download_model: downloading model")
        return Model.download_model_from_stub(
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
        _LOGGER.debug(f"download_model_from_stub: downloading model from stub {stub}")
        response_json = download_model_get_request(
            args=stub,
            file_name=None,
            force_token_refresh=force_token_refresh,
        )
        model = Model(
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
        sparse_name: Union[str, None] = None,
        sparse_category: Union[str, None] = None,
        sparse_target: Union[str, None] = None,
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
        _LOGGER.debug(
            f"search_models: searching models with args {args.model_url_args}"
        )
        response_json = search_model_get_request(
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

    def __repr__(self):
        return f"{self.__class__.__name__}(stub={self.stub})"

    def __str__(self):
        return f"{self.__class__.__name__}(stub={self.stub})"

    @property
    def is_base(self) -> bool:
        """
        :return: True if the model is a base model. Otherwise return False
        """
        return self.sparse_name == "base"

    @property
    def display_name(self) -> str:
        """
        :return: the display name for the model
        """
        return self._display_name

    @property
    def display_description(self) -> str:
        """
        :return: the description for the model
        """
        return self._display_description

    @property
    def recipes(self) -> List[Recipe]:
        """
        :return: list of recipes for the model
        """
        return self._recipes

    @property
    def original_recipe(self) -> Union[None, Recipe]:
        """
        :return: the original recipe used to create the model
        """
        original = None

        for recipe in self.recipes:
            if recipe.recipe_type_original:
                original = recipe
                break

        return original

    @property
    def transfer_learning_recipe(self) -> Union[None, Recipe]:
        """
        :return: the recipe to use for transfer learning from the model
        """
        transfer_learning = None

        for recipe in self.recipes:
            if recipe.recipe_type_transfer_learn:
                transfer_learning = recipe
                break

        return transfer_learning

    @property
    def data_inputs(self) -> Union[None, Data]:
        """
        :return: sample numpy data for inputs into the model
        """
        return self._data["inputs"] if "inputs" in self._data else None

    @property
    def data_outputs(self) -> Union[None, Data]:
        """
        :return: sample numpy data recorded as outputs from the model for a given input
        """
        return self._data["outputs"] if "outputs" in self._data else None

    @property
    def data_labels(self) -> Union[None, Data]:
        """
        :return: sample numpy data for labels for a given input
        """
        return self._data["labels"] if "labels" in self._data else None

    @property
    def data_originals(self) -> Union[None, Data]:
        """
        :return: sample numpy data as originals before any pre processing is applied
            to the data to create the inputs for the model
        """
        return self._data["originals"] if "originals" in self._data else None

    @property
    def data(self) -> Dict[str, Data]:
        """
        :return: A dictionary containing all of the Data objects for this model
        """
        return self._data

    @property
    def framework_files(self) -> List[File]:
        """
        :return: list of Files that are of type framework
        """
        return self._framework_files

    @property
    def onnx_files(self) -> List[File]:
        """
        :return: list of Files that are of type onnx
        """
        return self._onnx_files

    @property
    def onnx_file(self) -> Union[File, None]:
        """
        :return: The latest converted onnx file for the model
        """
        onnx_file = None

        for file in self.onnx_files:
            if file.file_type_onnx and file.display_name == "model.onnx":
                onnx_file = file

        return onnx_file

    @property
    def onnx_file_gz(self) -> Union[File, None]:
        """
        :return: The latest converted gziped onnx file for the model
        """
        onnx_file = None

        for file in self.onnx_files:
            if file.file_type_onnx_gz and file.display_name == "model.onnx":
                onnx_file = file

        return onnx_file

    @property
    def card_file(self) -> Union[File, None]:
        """
        :return: The markdown card representing info about the model
        """
        return self._card_file

    @property
    def tags(self) -> List[Tag]:
        """
        :return: a list of model repo tags for this model
        """
        return self._tags

    @property
    def results(self) -> List[Result]:
        """
        :return: a list of model repo results for this model
        """
        return self._results

    @property
    def release_version(self) -> Union[ReleaseVersion, None]:
        """
        :return: a model repo release version this model was released with
        """
        return self._release_version

    @property
    def user(self) -> Union[User, None]:
        """
        :return: the model repo user who uploaded this model
        """
        return self._user

    def data_loader(
        self, batch_size: int = 1, iter_steps: int = 0, batch_as_list: bool = True
    ) -> DataLoader:
        """
        Create a  data loader containing all of the available data for this model

        :param batch_size: the size of batches to create for the iterator
        :param iter_steps: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        :param batch_as_list: True to create the items from each dataset
            as a list, False for an ordereddict
        :return: The created data loader
        """
        data = self.data
        datasets = []

        for _, dat in data.items():
            datasets.append(dat.dataset())

        if len(datasets) < 1:
            raise FileNotFoundError(
                "no datasets available for this model to create a loader from"
            )

        return DataLoader(
            *datasets,
            batch_size=batch_size,
            iter_steps=iter_steps,
            batch_as_list=batch_as_list,
        )

    def sample_batch(
        self, batch_index: int = 0, batch_size: int = 1, batch_as_list: bool = True
    ) -> Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]:
        """
        Get a sample batch of data from the data loader

        :param batch_index: the index of the batch to get
        :param batch_size: the size of the batches to create the loader for
        :param batch_as_list: True to return multiple inputs/outputs/etc
            within the dataset as lists, False for an ordereddict
        :return: The sample batch for use with the model
        """
        loader = self.data_loader(batch_size=batch_size, batch_as_list=batch_as_list)

        return loader.get_batch(bath_index=batch_index)

    def download(
        self,
        overwrite: bool = False,
        refresh_token: bool = False,
        show_progress: bool = True,
    ):
        """
        Downloads a model repo file.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param refresh_token: refresh the auth token
        :param show_progress: True to use tqdm for progress, False to not show
        """
        _LOGGER.info(f"Downloading model {self.stub}")

        if self.card_file:
            _LOGGER.info(f"Downloading model card {self.stub}")
            self.card_file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        if self.onnx_file:
            _LOGGER.info(f"Downloading model onnx {self.stub}")
            self.onnx_file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        if self.onnx_file_gz:
            _LOGGER.info(f"Downloading model onnx gz {self.stub}")
            self.onnx_file_gz.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        self.download_framework_files(
            overwrite=overwrite,
            refresh_token=refresh_token,
            show_progress=show_progress,
        )

        for data in self._data.values():
            _LOGGER.info(f"Downloading model data {data.name} {self.stub}")
            data.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        for recipe in self._recipes:
            _LOGGER.info(f"Downloading model recipe {recipe.display_name} {self.stub}")
            recipe.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

    def download_framework_files(
        self,
        overwrite: bool = False,
        refresh_token: bool = False,
        show_progress: bool = True,
        extensions: Union[List[str], None] = None,
    ) -> List[str]:
        """
        Downloads the framework file(s) for this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param refresh_token: refresh the auth token
        :param show_progress: True to use tqdm for progress, False to not show
        :param extensions: List of file extensions to filter for. ex ['.pth', '.ptc'].
            If None or empty list, all framework files are downloaded. Default is None
        :return: List of paths to the downloaded files. Empty list if no files are
            found or matched
        """
        downloaded_paths = []
        for file in self._framework_files:
            if extensions and not any(
                file.display_name.endswith(ext) for ext in extensions
            ):  # skip files that do not end in valid extension
                continue

            _LOGGER.info(
                f"Downloading model framework file {file.display_name} {self.stub}"
            )
            file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )
            downloaded_paths.append(file.path)
        return downloaded_paths

    def search_similar_models(
        self,
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
    ) -> List["Model"]:
        """
        Search for similar models to this model

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
        _LOGGER.debug(f"search_similar_models: searching for models similar to {self}")
        return Model.search_models(
            domain=self.domain if match_domain else None,
            sub_domain=self.sub_domain if match_sub_domain else None,
            architecture=self.architecture if match_architecture else None,
            sub_architecture=self.sub_architecture if match_sub_architecture else None,
            framework=self.framework if match_framework else None,
            repo=self.repo if match_repo else None,
            dataset=self.dataset if match_dataset else None,
            training_scheme=self.training_scheme if match_training_scheme else None,
            sparse_name=self.sparse_name if match_sparse_name else None,
            sparse_category=self.sparse_category if match_sparse_category else None,
            sparse_target=self.sparse_target if match_sparse_target else None,
        )

    def search_sparse_models(
        self,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List["Model"]:
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
        _LOGGER.debug(
            f"search_sparse_models: searching for sparse models similar to {self}"
        )
        return [
            model
            for model in self.search_similar_models(
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
