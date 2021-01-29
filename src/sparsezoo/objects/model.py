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
from typing import Any, Dict, List, Union

import numpy

from sparsezoo.objects.data import Data
from sparsezoo.objects.downloadable import Downloadable
from sparsezoo.objects.file import File, FileTypes
from sparsezoo.objects.metadata import ModelMetadata
from sparsezoo.objects.optimization_recipe import OptimizationRecipe
from sparsezoo.objects.release_version import ReleaseVersion
from sparsezoo.objects.result import Result
from sparsezoo.objects.tag import Tag
from sparsezoo.objects.user import User
from sparsezoo.utils import DataLoader


__all__ = ["Model"]

_LOGGER = logging.getLogger(__name__)


class Model(Downloadable, ModelMetadata):
    """
    A model repo model

    :param display_name: the display name for the model
    :param display_description: the description for the model
    :param files: a list of model repo files for this model
    :param recipes: a list of model repo optimization recipes for this model
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
                OptimizationRecipe(
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

    def __repr__(self):
        return f"{self.__class__.__name__}(stub={self.stub})"

    def __str__(self):
        return f"{self.__class__.__name__}(stub={self.stub})"

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
    def recipes(self) -> List[OptimizationRecipe]:
        """
        :return: list of optimization recipes for the model
        """
        return self._recipes

    @property
    def original_recipe(self) -> Union[None, OptimizationRecipe]:
        """
        :return: the original optimization recipe used to create the model
        """
        original = None

        for recipe in self.recipes:
            if recipe.recipe_type_original:
                original = recipe
                break

        return original

    @property
    def transfer_learning_recipe(self) -> Union[None, OptimizationRecipe]:
        """
        :return: the optimization recipe to use for transfer learning from the model
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
