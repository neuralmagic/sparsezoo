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
import tarfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
import onnx

from sparsezoo.objects.directory import Directory
from sparsezoo.objects.file import File
from sparsezoo.utils import DataLoader, Dataset, load_numpy_list


__all__ = [
    "NumpyDirectory",
    "SelectDirectory",
    "OnnxGz",
    "AliasedSelectDirectory",
]

_LOGGER = logging.getLogger(__name__)

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
                _LOGGER.warning(
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
    :param tar_directory: optional pointer to the tar_directory
        of this directory. By default, when downloading the directory
        in question, we should download and extract the tarball.
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
        parent_directory: Optional[str] = None,
        stub_params: Optional[Dict[str, str]] = None,
        tar_directory: Optional[Directory] = None,
    ):
        self._default, self._available = None, None

        super().__init__(
            files=files,
            name=name,
            path=path,
            url=url,
            parent_directory=parent_directory,
        )
        self.tar_directory = tar_directory
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


class AliasedSelectDirectory(SelectDirectory):
    """
    A select directory that can be aliased to download a different
    file but still point to the same path. This is especially
    beneficial for cases where a tarball must be downloaded and extracted
    but the directory should point to one of the directory
    within the tarball.

    :param files: list of files contained within the SelectDirectory
    :param name: name of the SelectDirectory
    :param path: path of the SelectDirectory
    :param url: url of the SelectDirectory
    :param parent_directory: path of the parent SelectDirectory
    :param stub_params: dictionary of zoo stub params that this directory
        was specified with
    :param download_alias: name of the file to download
    """

    def __init__(self, *args, download_alias: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.download_alias = download_alias

    @contextmanager
    def _override_name(self):
        """
        A context manager to temporarily override the name of the directory
        to the download alias.
        """
        old_name = self.name
        if self.download_alias:
            self.name = self.download_alias
        yield
        self.name = old_name

    def download(self, *args, **kwargs):
        """
        Override the download method to temporarily override the name
        of the directory to the download alias.
        """
        with self._override_name():
            return super().download(*args, **kwargs)

    @property
    def path(self):
        """
        Override the path property to download and extract
        the download alias file but point to the path of the
        actual expected file

        :raises FileNotFoundError: if the expected file does not
            exist, (it wasn't downloaded or extracted)
        :return: path to the expected file
        """
        super().path
        actual_file_path = (
            Path(self._path).with_name(self.name)
            if self._path.endswith(self.download_alias)
            else Path(self._path)
        )
        if not actual_file_path.exists():
            raise FileNotFoundError(
                f"The directory {actual_file_path} should have been "
                "downloaded but does not exist"
            )
        return str(actual_file_path)


class OnnxGz(Directory):
    """
    Special class to handle onnx.model.tar.gz files.
    Desired behavior is that all information about files included in the tarball are
    available however, when the `path` property is accessed, it will point only
    to the `model.onnx` as this is the expected behavior for loading an onnx model
    with or without external data.

    Class Invariants:
        - `self.name` attribute of this class will point to the name of the onnx file
        - `self._path` and `self.path` will point to the path of the extracted
            onnx model
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"OnnxGz(name={self.name}, path={self._path})"

    @property
    def path(self):
        """
        Assumptions:
            - the tarball must contain atleast one `model.onnx` file
            - the tarball may or may not contain additional external data or
                onnx model file(s)
            - the tarball will be extracted to the same directory as the tarball

        :post-condition: self._path will point to the path of the extracted
            onnx model, and it exists
        :return: path to the onnx model
        """

        expected_path: Path = (
            Path(self._path)
            if self._path is not None
            else Path(self.parent_directory) / "model.onnx.tar.gz"
        )

        # _path can also point to parent directory of the tarball when
        #  this object is initialized from files
        if expected_path.is_dir():
            # point expected to model.onnx.tar.gz
            expected_path = expected_path / "model.onnx.tar.gz"

        # point path to model.onnx.tar.gz before download/unzip
        self._path = str(expected_path)

        if not expected_path.exists():
            # download the tarball if it does not exist
            self.download()

        self._check_if_extracted()
        if self.is_archive:
            # if the tarball is not extracted, extract it
            self.unzip()

        onnx_model_path = expected_path.with_name(name="model.onnx")
        if not onnx_model_path.exists():
            raise FileNotFoundError(
                f"Expected to find model.onnx at {onnx_model_path}, "
                "but it does not exist."
            )

        # point _path to model.onnx
        self._path = str(onnx_model_path)
        return self._path

    def _check_if_extracted(self):
        """
        set `is_archive` to False if the tarball is extracted, else set it to
        True. Condition for being extracted is that all members of the tarball
        are extracted and exist as files in the same directory as the tarball.
        """
        # expected file will point to the tarball
        expected_path: Path = (
            Path(self._path)
            if self._path is not None
            else Path(self.parent_directory) / "model.onnx.tar.gz"
        )

        if expected_path.is_dir():
            # point expected to model.onnx.tar.gz
            expected_path = expected_path / "model.onnx.tar.gz"

        model_gz_path = expected_path.with_name(name="model.onnx.tar.gz")
        # assert all members of  model.onnx.tar.gz have been extracted
        for zipped_filename in tarfile.open(model_gz_path).getnames():
            unzipped_file_path = expected_path.with_name(zipped_filename)
            if not unzipped_file_path.exists():
                _LOGGER.debug(f"{unzipped_file_path} does not exist, was it extracted?")
                self.is_archive = True
                return
        self.is_archive = False
