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
from sparsezoo.v2.directory import Directory
from sparsezoo.v2.file import File


__all__ = ["NumpyDirectory", "SampleOriginals"]

NUMPY_DIRECTORY_NAMES = ["sample-inputs", "sample-outputs"]


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