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
from typing import List, Optional

import onnx

from sparsezoo.utils.numpy import load_numpy_list
from sparsezoo.v2.objects.directory import Directory
from sparsezoo.v2.objects.file import File


__all__ = ["NumpyDirectory"]

NUMPY_DIRECTORY_NAMES = ["sample_inputs", "sample_outputs"]


class NumpyDirectory(Directory):
    """
    Object that represents a directory with unedited data
    that can be used as inputs to a training pipeline.

    :param files: list of files contained within the NumpyDirectory
    :param name: name of the NumpyDirectory
    :param path: path of the NumpyDirectory
    :param url: url of the NumpyDirectory
    :param owner_path: path of the parent NumpyDirectory
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
        owner_path: Optional[str] = None,
    ):
        super().__init__(
            files=files, name=name, path=path, url=url, owner_path=owner_path
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
    :param name: name of the NumpyDirectory
    :param path: path of the NumpyDirectory
    :param url: url of the NumpyDirectory
    :param owner_path: path of the parent SelectDirectory
    """

    def __init__(
        self,
        files: List[File],
        name: str,
        path: Optional[str] = None,
        url: Optional[str] = None,
        owner_path: Optional[str] = None,
    ):
        self._default, self._available = None, None

        super().__init__(
            files=files, name=name, path=path, url=url, owner_path=owner_path
        )

        self.files_dict = self.files_to_dictionary()

    def __getitem__(self, key):
        file = self.files_dict[key]
        file.get_path()
        return file

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
        if self.name == "recipe" and "original" in self.files_dict:
            return self["original"]
        elif self.name == "training" and "preqat" in self.files_dict:
            return self["preqat"]
        else:
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
