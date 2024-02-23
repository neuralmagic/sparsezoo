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
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import onnx
from onnx import ModelProto, TensorProto
from onnx.external_data_helper import ExternalDataInfo

from sparsezoo.utils.helpers import clean_path


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "onnx_includes_external_data",
    "save_onnx",
    "validate_onnx",
    "load_model",
    "split_external_data",
    "EXTERNAL_ONNX_DATA_NAME",
]


EXTERNAL_ONNX_DATA_NAME = "model.data"

# DUMP_EXTERNAL_DATA_TRESHOLD is a limiting value
# for the model saved with external data. If the model
# is larger than this value, it will be saved with external data.
# The threshold is expressed in bits and corresponds
# set to 500MB. This is roughly the size of
# 250 million parameters (assuming fp16).
DUMP_EXTERNAL_DATA_THRESHOLD = 4e9


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
    max_external_file_size: int = 16e9,
    external_data_file: Optional[str] = None,
    do_split_external_data: bool = True,
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
    :param max_external_file_size: The maximum file size in bytes of a single split
        external data out file. Defaults to 16000000000 (16e9 = 16GB)
    :param do_split_external_data: True to split external data file into chunks of max
    size max_external_file_size, false otherwise
    :return True if the model was saved with external data, False otherwise.
    """
    if external_data_file is not None:
        _LOGGER.debug(
            f"Saving with external data, with file chunks of maximum size "
            f"{max_external_file_size / 1e9} GB"
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
        if do_split_external_data:
            split_external_data(model_path, max_file_size=max_external_file_size)
        return True

    if model.ByteSize() > DUMP_EXTERNAL_DATA_THRESHOLD:
        external_data_file = external_data_file or EXTERNAL_ONNX_DATA_NAME
        _LOGGER.debug(
            "The ONNX model is too large to be saved as a single protobuf. "
            "Saving with external data, with file chunks of maximum size "
            f"{max_external_file_size / 1e9} GB"
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
        if do_split_external_data:
            split_external_data(model_path, max_file_size=max_external_file_size)
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


def load_model(
    model: Union[str, ModelProto, Path], load_external_data: bool = True
) -> ModelProto:
    """
    Load an ONNX model from an onnx model file path. If a ModelProto
    is given, then it is returned.

    :param model: the model proto or path to the model ONNX file to check for loading
    :param load_external_data: if a path is given, whether or not to also load the
        external model data
    :return: the loaded ONNX ModelProto
    """
    if isinstance(model, ModelProto):
        return model

    if isinstance(model, (Path, str)):
        return onnx.load(clean_path(model), load_external_data=load_external_data)

    raise TypeError(f"unknown type given for model: {type(model)}")


def split_external_data(
    model_path: str,
    max_file_size: int = 16e9,
    allow_large_tensors: bool = True,
):
    """
    Splits the external_data_path file into multiple files of size no larger than
    max_file_size. ONNX model will be updated in-place, external data file will be
    replaced with multiple files in the same location with the same base name

    :param model_path: path to ONNX model file who has external data writen
        to a single file in the same directory
    :param max_file_size: maximum file size in bytes of a single split out file.
        defaults to 16000000000 (16e9 = 16GB)
    :param allow_large_tensors: if False, will raise an exception if any model tensor
        is larger than max_file_size. If True, will write the large tensor to a single
        file regardless of max_file_size. Default True
    :raises ValueError: if the given model does not have external data
    :raises ValueError: if the given model has external data written to multiple
        locations
    :raises RuntimeError: if the external data file does not exist in the same
        directory as the model
    """
    model = onnx.load(model_path, load_external_data=False)
    base_dir = str(Path(model_path).parent)

    external_data_info_by_name = {
        init.name: ExternalDataInfo(init)
        for init in model.graph.initializer
        if init.external_data
    }

    # VALIDATION: model has external data written to a single file in the same directory
    if not external_data_info_by_name:
        raise ValueError(f"{model_path} does not contain external data")

    external_data_files = {
        info.location for info in external_data_info_by_name.values()
    }
    if len(external_data_files) > 1:
        raise ValueError(
            f"External data files found: {external_data_files} for model "
            f"{model_path}. External data must be written to a single file to split"
        )

    external_data_file = external_data_files.pop()
    external_data_file_path = os.path.join(base_dir, external_data_file)
    if not os.path.exists(external_data_file_path):
        raise RuntimeError(
            f"{external_data_file_path} not found. {model_path} must have external "
            "data written to a single file in the same directory"
        )
    if os.path.getsize(external_data_file_path) <= max_file_size:
        # return immediately if file is small enough to not split
        return

    # UPDATE: external data info of graph tensors so they point to the new split out
    # files with updated offsets
    current_external_data_file_number = 1
    current_external_data_file_size = 0  # bytes
    new_files_to_old_byte_ranges = defaultdict(list)  # values: (start_byte, num_bytes)
    for init in model.graph.initializer:
        if init.name not in external_data_info_by_name:
            continue  # not external data tensor
        info = external_data_info_by_name[init.name]
        tensor_size = info.length

        if not allow_large_tensors and tensor_size > max_file_size:
            raise ValueError(
                f"tensor {init.name} has size {tensor_size} greater than max allowed "
                f"size {max_file_size}. Set allow_large_tensors=True to allow"
            )

        if tensor_size + current_external_data_file_size > max_file_size:
            # writing this tensor would set the current file over the max size, start
            # a new file
            current_external_data_file_number += 1
            current_external_data_file_size = 0

        # update the file of the tensor and its offset for the new data file
        updated_location = f"{external_data_file}.{current_external_data_file_number}"
        _set_external_data(
            tensor=init,
            location=updated_location,
            offset=current_external_data_file_size,
            length=info.length,
        )
        current_external_data_file_size += info.length

        # add bytes to the current file to be written
        new_files_to_old_byte_ranges[updated_location].append(
            (info.offset, info.length)
        )

    # WRITE - new data files
    with open(external_data_file_path, "rb") as external_data_file_reader:
        for updated_file_name, tensor_ranges in new_files_to_old_byte_ranges.items():
            updated_file_path = os.path.join(base_dir, updated_file_name)
            _write_external_data_file_from_base_bytes(
                updated_file_path, tensor_ranges, external_data_file_reader
            )

    # DELETE - old external data file
    os.remove(external_data_file_path)

    # WRITE - ONNX model with updated tensor external data info
    onnx.save(model, model_path)


def _write_external_data_file_from_base_bytes(
    new_file_name, original_byte_ranges, original_file_bytes_reader
):
    # original_byte_ranges: List[(int, int)] must be in order of offset
    with open(new_file_name, "wb") as new_file_writer:
        for original_data_start, original_data_length in original_byte_ranges:
            # set reader to start of a tensor
            original_file_bytes_reader.seek(original_data_start)
            # read entire tensor
            tensor_bytes = original_file_bytes_reader.read(original_data_length)
            # write tensor to new file
            new_file_writer.write(tensor_bytes)


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


def _set_external_data(
    tensor: TensorProto,
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
) -> None:
    # ADAPTED FROM: https://github.com/onnx/onnx/blob/e724cc33616ff614dd8555743e9d707b5a7c5492/onnx/external_data_helper.py#L80  # noqa: E501
    # adapted to skip blocking validation checks not relevant to our use case
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in {
        "location": location,
        "offset": int(offset) if offset is not None else None,
        "length": int(length) if length is not None else None,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)
