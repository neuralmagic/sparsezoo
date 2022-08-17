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
Class objects for standardization and validation of a model folder structure
"""


import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import numpy
import onnx

from sparsezoo.objects.directory import Directory
from sparsezoo.objects.file import File
from sparsezoo.utils import DataLoader, Dataset, load_numpy_list


__all__ = ["NumpyDirectory", "SelectDirectory"]

NUMPY_DIRECTORY_NAMES = ["sample_inputs", "sample_outputs"]


class NumpyDirectory(Directory):
    """
    Object that represents a directory with unedited data
    that can be used as inputs to a training pipeline.

    :param files: list of files contained within the NumpyDirectory
    :param name: name of the NumpyDirectory
    :param path: path of the NumpyDirectory
    :param url: url of the NumpyDirectory
    :param parent_directory: path of the parent NumpyDirectory
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
        parent_directory: Optional[str] = None,
    ):
        super().__init__(
            files=files,
            name=name,
            path=path,
            url=url,
            parent_directory=parent_directory,
        )

    def validate(
        self,
        model: Optional[onnx.ModelProto] = None,
        integration: Optional[str] = None,
        strict_mode: bool = True,
    ) -> bool:
        """
        Validate whether all the files in numpy directory
        adhere to the convention imposed by the integration.

        :param model: ONNX model
        :param integration: integration: name of the specific integration
            (e.g. transformers, YOLOv5 etc.)
        :param strict_mode:
            - if strict_mode: validation will raise ValueError on error
            - if not strict_mode: validation will raise warning on error
        :return: boolean flag; True if files are valid and no errors arise
        """

        for file in self.files:
            file._validate_numpy(strict_mode=strict_mode)

        if model:
            if not self._validate_model(model):
                logging.warning(
                    "Could not validate an NumpyDirectory given the provided onnx model."  # noqa: E501
                )
                return False

        return True

    def __iter__(self) -> OrderedDict:
        for file in self.files:
            for numpy_dict in load_numpy_list(file.path):
                yield numpy_dict

    def dataset(self) -> Dataset:
        """
        A dataset for interacting with the sample data.
        If the data is not found on the local disk, will automatically download.

        :return: The created dataset from the sample data files
        """

        # sample_{...} or sample_{...}.tar.gz --> sample_{...}
        dataset_name = self.name.split(".")[0]
        return Dataset(dataset_name, self.path)

    def loader(
        self, batch_size: int = 1, iter_steps: int = 0, batch_as_list: bool = True
    ) -> DataLoader:
        """
        A dataloader for interfacing with the sample data in a batched format.

        :param batch_size: the size of the batches to create the loader for
        :param iter_steps: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        :param batch_as_list: True to return multiple inputs/outputs/etc
            within the dataset as lists, False for an ordereddict
        :return: The created dataloader from the sample data files
        """
        return DataLoader(
            self.dataset(),
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
        loader = self.loader(batch_size=batch_size, batch_as_list=batch_as_list)

        return loader.get_batch(bath_index=batch_index)

    def _validate_model(self, model: onnx.ModelProto) -> bool:
        file_name = self.name.split(".")[:-2] if self.is_archive else self.name
        if file_name not in NUMPY_DIRECTORY_NAMES:
            raise ValueError(
                "Expected the name of NumpyDirectory to be in "
                f"{NUMPY_DIRECTORY_NAMES + [x + '.tar.gz' for x in NUMPY_DIRECTORY_NAMES]}. "  # noqa E501
                f"Found name: {self.name}."
            )
        validating_inputs = file_name == "sample_inputs"
        expected_names = (
            [inp.name for inp in model.graph.input]
            if validating_inputs
            else [out.name for out in model.graph.output]
        )
        for file in self.files:
            numpy_list = load_numpy_list(file.path)
            for numpy_dict in numpy_list:
                if expected_names != list(numpy_dict.keys()):
                    key_type = "input" if validating_inputs else "output"
                    raise ValueError(
                        f"The {key_type} keys in npz dictionary do not match "
                        f"the {key_type} names of the provided model.\n"
                        f"Expected keys from onnx model: {expected_names}.\n"
                        f"Found keys in numpy {key_type}: {list(numpy_dict.keys())}."
                    )
        return True


class SelectDirectory(Directory):
    """
    Object that represents a directory from which the
    user can fetch the contained directories and file
    by key values.

    :param files: list of files contained within the SelectDirectory
    :param name: name of the SelectDirectory
    :param path: path of the SelectDirectory
    :param url: url of the SelectDirectory
    :param parent_directory: path of the parent SelectDirectory
    :param stub_params: dictionary of zoo stub params that this directory
        was specified with
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
        parent_directory: Optional[str] = None,
        stub_params: Optional[Dict[str, str]] = None,
    ):
        self._default, self._available = None, None

        super().__init__(
            files=files,
            name=name,
            path=path,
            url=url,
            parent_directory=parent_directory,
        )

        self._stub_params = stub_params or {}
        self.files_dict = self.files_to_dictionary()

    def __getitem__(self, key):
        file = self.files_dict[key]
        return file

    @property
    def stub_params(self) -> Dict[str, str]:
        """
        :return: mapping of variable name to value for query params in zoo stub
            this directory was initialized from
        """
        return self._stub_params

    def files_to_dictionary(self):
        if self.name == "recipe":
            recipe_dictionary = {
                file.name.replace("recipe_", "").replace(".md", ""): file
                for file in self.files
            }
            return recipe_dictionary
        elif self.name == "training":
            training_dictionary = {"preqat": self}
            return training_dictionary
        elif self.name == "deployment":
            deployment_dictionary = {"default": self}
            return deployment_dictionary

    @property
    def default(self):
        if self.name == "recipe":
            # maybe to extract desired recipe name from stub params
            recipe_type = self.stub_params.get("recipe_type") or (
                self.stub_params.get("recipe")
            )
            if not recipe_type and "original" in self.files_dict:
                # default to original recipe
                return self["original"]
            # try to find recipe satisfying the recipe type
            for recipe_name in self.files_dict:
                if recipe_type and recipe_type.lower() == recipe_name.lower():
                    return self[recipe_name]
        if self.name == "training" and "preqat" in self.files_dict:
            return self["preqat"]
        # default to first value
        return next(iter(self.files_dict.values()))

    @default.setter
    def default(self, value):
        self._default = value

    @property
    def available(self):
        return list(self.files_dict.keys())

    @available.setter
    def available(self, value):
        self._available = value
