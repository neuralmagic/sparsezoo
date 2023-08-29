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

import logging
import os
from typing import Optional, Union

import onnx
from onnx import ModelProto

from sparsezoo.utils.helpers import clean_path


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "onnx_includes_external_data",
    "save_onnx",
    "validate_onnx",
    "load_model",
    "EXTERNAL_ONNX_DATA_NAME",
]


EXTERNAL_ONNX_DATA_NAME = "model.data"


def onnx_includes_external_data(model: ModelProto) -> bool:
    """
    Check whether the ModelProto in memory includes the external
    data or not.

    If the model.onnx does not contain the external data, then the
    initializers of the model are pointing to the external data file
    (they are not empty)

    :param model: the ModelProto to check
    :return True if the model was loaded with external data, False otherwise.
    """

    initializers = model.graph.initializer

    is_data_saved_to_disk = any(
        initializer.external_data for initializer in initializers
    )
    is_data_included_in_model = not is_data_saved_to_disk

    return is_data_included_in_model


def save_onnx(
    model: ModelProto,
    model_path: str,
    external_data_file: Optional[str] = None,
) -> bool:
    """
    Save model to the given path.

    If the model's size is larger than the maximum protobuf size:
        -   it will be saved with external data
    If the model's size is smaller than the maximum protobuf size:
        -   and the user nevertheless specifies 'external_data_file',
            the model will be saved with external data.

    :param model: The model to save.
    :param model_path: The path to save the model to.
    :param external_data_file: The optional name save the external data to. Must be
        relative to the directory `model` is saved to. If the model is too
        large to be saved as a single protobuf, and this argument is None,
        the external data file will be coerced to take the default name
        specified in the variable EXTERNAL_ONNX_DATA_NAME
    :return True if the model was saved with external data, False otherwise.
    """
    if external_data_file is not None:
        _LOGGER.debug(f"Saving with external data: {external_data_file}")
        _check_for_old_external_data(
            model_path=model_path, external_data_file=external_data_file
        )
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_file,
        )
        return True

    if model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
        external_data_file = external_data_file or EXTERNAL_ONNX_DATA_NAME
        _LOGGER.warning(
            "The ONNX model is too large to be saved as a single protobuf. "
            f"Saving with external data: {external_data_file}"
        )
        _check_for_old_external_data(
            model_path=model_path, external_data_file=external_data_file
        )
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_file,
        )
        return True

    onnx.save(model, model_path)
    return False


def validate_onnx(model: Union[str, ModelProto]):
    """
    Validate that a file at a given path is a valid ONNX model.
    Raises a ValueError if not a valid ONNX model.

    :param model: the model proto or path to the model
        ONNX file to check for validation
    """
    try:
        onnx_model = load_model(model)
        if onnx_model.ByteSize() > onnx.checker.MAXIMUM_PROTOBUF:
            if isinstance(model, str):
                onnx.checker.check_model(model)
            else:
                _LOGGER.warning(
                    "Attempting to validate an in-memory ONNX model with "
                    f"size > {onnx.checker.MAXIMUM_PROTOBUF} bytes."
                    "`validate_onnx` skipped, as large ONNX models cannot "
                    "be validated in-memory. To validate this model, save "
                    "it to disk and call `validate_onnx` on the file path."
                )
            return
        onnx.checker.check_model(onnx_model)
    except Exception as err:
        if not onnx_includes_external_data(model):
            _LOGGER.warning(
                "Attempting to validate an in-memory ONNX model "
                "that has been loaded without external data. "
                "This is currently not supported by the ONNX checker. "
                "The validation will be skipped."
            )
            return
        raise ValueError(f"Invalid onnx model: {err}")


def load_model(model: Union[str, ModelProto]) -> ModelProto:
    """
    Load an ONNX model from an onnx model file path. If a ModelProto
    is given, then it is returned.

    :param model: the model proto or path to the model ONNX file to check for loading
    :return: the loaded ONNX ModelProto
    """
    if isinstance(model, ModelProto):
        return model

    if isinstance(model, str):
        return onnx.load(clean_path(model))

    raise ValueError(f"unknown type given for model: {type(model)}")


def _check_for_old_external_data(model_path: str, external_data_file: str):
    old_external_data_file = os.path.join(
        os.path.dirname(model_path), external_data_file
    )
    if os.path.exists(old_external_data_file):
        _LOGGER.warning(
            f"Attempting to save external data for a model: {model_path} "
            f"to a directory:{os.path.dirname(model_path)} "
            f"that already contains external data file: {external_data_file}. "
            "The external data file will be overwritten."
        )
        os.remove(old_external_data_file)

    return
