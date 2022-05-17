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
import re
from collections import OrderedDict
from typing import List, Optional

import onnx

from sparsezoo.refactor.directory import Directory
from sparsezoo.refactor.file import File
from sparsezoo.utils.numpy import load_numpy_list


__all__ = ["FrameworkFiles", "NumpyDirectory", "SampleOriginals"]

NUMPY_DIRECTORY_NAMES = ["sample-inputs", "sample-outputs"]


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

        self.valid_nested_folder_patterns = ["checkpoint_(.*)/", "logs/"]

    # TODO: Add support for model cards (pull integration from
    #  model.md and specify which files to validate)
    def validate(self, integration: Optional[str] = None) -> bool:
        """
        Validates the structure of framework files.

        :param integration: integration: name of the specific integration
            (e.g. transformers, YOLOv5 etc.)
        :return: boolean flag; True if files are valid and no errors arise
        """
        validations = {}
        for file in self.files:
            if isinstance(file, Directory):
                # check for Directory files
                validations[file.name] = self._check_directory(directory=file)
            else:
                # check for File files
                validations[file.name] = self._check_file(file=file)

        if not all(validations.values()):
            logging.warning(
                f"Following files: "
                f"{[key for key, value in validations.items() if not value]} "
                "were not successfully validated."
            )

        return all(validations.values())

    def get_file(self, file_name: str) -> Optional[File]:
        """
        Fetch a file from the FrameworkFiles by name.

        :param file_name: name of the file to be fetched
        :return: File if found, otherwise None
        """
        for file in self.files:
            if isinstance(file, Directory):
                if file.name == file_name:
                    return file
                else:
                    file = file.get_file(file_name=file_name)
                    if file:
                        return file

            elif file.name == file_name:
                return file

        logging.warning(f"File with name {file_name} not found!")
        return None

    def _check_valid_folder_name(self, file: File) -> bool:
        # checks that any nested folders’ names
        # either follow `checkpoint_id/` or `logs/`
        checks = [False for _ in self.valid_nested_folder_patterns]
        for i, pattern in enumerate(self.valid_nested_folder_patterns):
            pattern = re.compile(pattern)
            match = re.search(pattern, file.path)
            if match:
                checks[i] = True
        if not any(checks):
            raise ValueError(
                f"File: {file.name} has path {file.path}, which does "
                "not include any of the following "
                f"directories: {self.valid_nested_folder_patterns}."
            )
        return True

    def _check_file(self, file: File) -> bool:
        # TODO: Assuming for now that all files are loadable,
        # this may not be the case when we include integrations
        self._check_valid_folder_name(file)
        return file.validate()

    def _check_directory(self, directory: Directory) -> bool:
        if not directory.files and not directory._is_tar():
            raise ValueError(f"Detected empty directory: {directory.name}!")
        validations = []
        for file in directory.files:
            if isinstance(file, Directory):
                # nested directory
                validations.append(self._check_directory(file))
            else:
                validations.append(self._check_file(file=file))
        return all(validations)


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

    def __iter__(self) -> File:
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

    def _validate_model(self, model: onnx.ModelProto) -> bool:
        file_name = self.name.split(".")[:-2] if self.is_archive else self.name
        if file_name not in NUMPY_DIRECTORY_NAMES:
            raise ValueError(
                "Expected the name of NumpyDirectory to be in "
                f"{NUMPY_DIRECTORY_NAMES + [x + '.tar.gz' for x in NUMPY_DIRECTORY_NAMES]}. "  # noqa E501
                f"Found name: {self.name}."
            )
        validating_inputs = file_name == "sample-inputs"
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
