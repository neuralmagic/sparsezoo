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

import json
import logging
import os
import pathlib
import re
import tarfile
from typing import List, Optional, Union

import onnx
import yaml

from PIL import Image
from src.sparsezoo.utils.numpy import load_numpy_list


class File:
    """
    Object to wrap around common files. Currently, supporting:
    - numpy files
    - onnx files
    - yaml files
    - markdown files
    - json files
    - csv files
    - image files

    :param name: name of the File
    :param path: path of the File
    :param url: url of the File
    """

    def __init__(
        self, name: str, path: Optional[str] = None, url: Optional[str] = None
    ):
        self.name = name
        self.url = url
        self.path = path

        # self.path can have any extension, including no extension.
        # However, the File object also contains information
        # About its loadable extensions.
        # Loadable files can be read into the memory.
        self.loadable_extensions = {
            ".npz": self._validate_numpy,
            ".onnx": self._validate_onnx,
            ".yaml": self._validate_yaml,
            ".md": self._validate_markdown,
            ".json": self._validate_json,
            ".csv": self._validate_csv,
            ".jpg": self._validate_img,
            ".png": self._validate_img,
            ".jpeg": self._validate_img,
        }

    # TODO: Add support for various integrations
    def validate(
        self, strict_mode: bool = True, integration: Optional[str] = None
    ) -> bool:
        """
        Validate whether the File object is loadable or not.

        :param strict_mode: specifies the behavior of private `_validate_{}` methods:
            - if strict_mode: method will raise ValueError on error
            - if not strict_mode: method will raise warning on
                error
        :param integration: name of the specific integration
            (e.g. transformers, YOLOv5 etc.)
        :return: boolean flag; True if File instance is loadable, otherwise False
        """
        _, extension = os.path.splitext(self.path)

        if extension in self.loadable_extensions.keys():
            validation_function = self.loadable_extensions[extension]
            validation_function(strict_mode=strict_mode)
            return True

        else:
            return False

    def _validate_numpy(self, strict_mode):
        if not load_numpy_list(self.path):
            self._throw_error(
                error_msg="Numpy file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_onnx(self, strict_mode):
        if not onnx.load(self.path):
            self._throw_error(
                error_msg="Onnx file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_yaml(self, strict_mode):
        try:
            with open(self.path) as file:
                yaml.load(file, Loader=yaml.FullLoader)
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Yaml file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_markdown(self, strict_mode):
        try:
            with open(self.path) as file:
                file.readlines()
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Markdown file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_json(self, strict_mode):
        try:
            with open(self.path) as file:
                json.load(file)
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Json file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_csv(self, strict_mode):
        try:
            with open(self.path) as file:
                file.readlines()
        except Exception as error:  # noqa: F841
            self._throw_error(
                error_msg="Csv file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _validate_img(self, strict_mode):
        if not Image.open(self.path):
            self._throw_error(
                error_msg="Image file could not been loaded properly",
                strict_mode=strict_mode,
            )

    def _throw_error(self, error_msg, strict_mode):
        if strict_mode:
            raise ValueError(error_msg)
        else:
            logging.warning(error_msg)


class Directory(File):
    """
    Object that represents a directory.

    :param files: list of files contained within the Directory
    :param name: name of the Directory
    :param path: path of the Directory
    :param url: url of the Directory
    """

    def __init__(
        self,
        name: str,
        files: Optional[List[File]] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):

        self.files = files

        if path is None:
            path = self._infer_path_from_files()
        super().__init__(name=name, path=path, url=url)

    def gzip(self) -> str:
        """
        Create a tar archive file from the Directory.
        The tar archive file would contain all the files
        in the Directory object.
        The tar archive file would be saved in the parent
        directory of the Directory object.
        :return: path to the tar archive file created
        """
        if self.path is None:
            raise ValueError(
                "Attempting to zip the Directory object files, "
                "but `self.path` is None. "
                "Class object requires pointer to parent "
                "folder directory to know where to save the tar archive file."
            )
        parent_path = pathlib.PurePath(self.path).parent
        tar_file_path = os.path.join(parent_path, self.name + ".tar.gz")
        with tarfile.open(tar_file_path, "w") as tar:
            for file in self.files:
                tar.add(file.path)
        return tar_file_path

    def unzip(self, path_to_extract: Optional[str] = None) -> bool:
        """
        If Directory is a tar archive, extract it to:
         - parent directory of the Directory object (by default)
         - specified path (if `tar_file_path` provided).

        Note: `path_to_extract` is relative to the working dir!

        :param path_to_extract: path where the tar archive file is extracted to
        :return boolean flag; True if archive is properly extracted
        """

        if path_to_extract is None:
            path_to_extract = "."

        _, *extension = self.path.split(".")
        if extension != ["tar", "gz"]:
            raise ValueError(
                "Attempting to extract tar archive, but the extension "
                f"of the Directory object is {extension}, not `.tar.gz`"
            )

        tar = tarfile.open(self.path, "r")
        tar.extractall(path=path_to_extract)
        tar.close()
        return True

    def __len__(self):
        return len(self.files)

    def _infer_path_from_files(self) -> Union[str, None]:
        # if Directory object describes a local directory, we may try to infer the path
        # from the attributes of the files contained within
        paths = [os.path.dirname(file.path) for file in self.files]

        # assert that all the files have the same dirname.
        if len(set(paths)) == 1:
            return paths[0]
        else:
            return None


class FrameworkFiles(Directory):
    """
    Object that represents a directory with framework files.

    :param files: list of files contained within the FrameworkFiles
    :param name: name of the FrameworkFiles
    :param path: path of the FrameworkFiles
    :param url: url of the FrameworkFiles
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):

        super().__init__(files=files, name=name, path=path, url=url)

        self.nested_folders_names = ["checkpoint_(.*)/", "logs/"]

    # TODO: Add support for model cards (pull integration from
    #  model.md and specify which files to validate)
    def validate(self, integration: Optional[str] = None) -> bool:
        """
        Validate whether all the framework files adhere
        to the convention imposed by the integration.

        :param integration: integration: name of the specific integration
            (e.g. transformers, YOLOv5 etc.)
        :return: boolean flag; True if files are valid and no errors arise
        """
        for file in self.files:
            if isinstance(file, Directory):
                # check for Directory files
                self._check_directory(directory=file)
            else:
                # check for File files
                self._check_file(file=file)

        return True

    def get_file(self, file_name: str) -> Optional[File]:
        """
        Fetch a file from the FrameworkFiles by name.

        :param file_name: name of the file to be fetched
        :return: File if found, otherwise None
        """
        for file in self.files:
            if file.name == file_name:
                file_found = file

            elif isinstance(file, Directory):
                file_found = file.get_file(file_name=file_name)

            else:
                logging.warning(f"File with name {file_name} not found!")
                file_found = None

        return file_found

    def _check_valid_folder_name(self, file: File) -> bool:
        # checks that any nested folders’ names
        # either follow `checkpoint_id/` or `logs/`
        checks = [False for _ in self.nested_folders_names]
        for i, pattern in enumerate(self.nested_folders_names):
            pattern = re.compile(pattern)
            match = re.search(pattern, file.path)
            if match:
                checks[i] = True
        if not any(checks):
            raise ValueError(
                f"File: {file.name} has path {file.path}, which does "
                "not include any of the following "
                f"directories: {self.nested_folders_names}."
            )
        return True

    def _check_file(self, file: File) -> bool:
        # TODO: Assuming for now that all files are loadable,
        #  this may not be the case when we include integrations
        self._check_valid_folder_name(file)
        file.validate()

    def _check_directory(self, directory: Directory) -> bool:
        if not directory.files:
            raise ValueError(f"Detected empty directory: {directory.name}!")
        for file in directory.files:
            if isinstance(file, Directory):
                # nested directory
                self._check_directory(file)
            else:
                for file in directory.files:
                    self._check_file(file=file)
        return True


class SampleOriginals(Directory):
    """
    Object that represents a directory with unedited data that
    can be used as inputs to a training pipeline.

    :param files: list of files contained within the SampleOriginals
    :param name: name of the SampleOriginals
    :param path: path of the SampleOriginals
    :param url: url of the SampleOriginals
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):
        super().__init__(files=files, name=name, path=path, url=url)

    def validate(self, integration: Optional[str] = None) -> bool:
        """
        Validate whether all the sample originals files adhere to
        the convention imposed by the integration.

        :param integration: name of the specific integration
            (e.g. transformers, YOLOv5 etc.)
        :return: boolean flag; True if files are valid and no errors arise
        """
        # TODO: Assuming for now that all files are loadable
        for file in self.files:
            file.validate()

        return True

    def __iter__(self):
        for file in self.files:
            yield file


class NumpyDirectory(Directory):
    """
    Object that represents a directory with unedited data
    that can be used as inputs to a training pipeline.

    :param files: list of files contained within the NumpyDirectory
    :param name: name of the NumpyDirectory
    :param path: path of the NumpyDirectory
    :param url: url of the NumpyDirectory
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
    ):
        super().__init__(files=files, name=name, path=path, url=url)

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
            self._validate_model(model)

        return True

    def _validate_model(self, model: Optional[onnx.ModelProto]):
        input_names = [inp.name for inp in model.graph.input]
        for file in self.files:
            numpy_list = load_numpy_list(file.path)
            for numpy_dict in numpy_list:
                if input_names != list(numpy_dict.keys()):
                    raise ValueError(
                        "The keys in npz dictionary do not match "
                        "the input names of the provided model!"
                    )

    def __iter__(self):
        for file in self.files:
            for numpy_dict in load_numpy_list(file.path):
                yield numpy_dict
