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
Code related to numpy array files
"""

import glob
import logging
import os
import tarfile
from collections import OrderedDict
from io import BytesIO
from typing import Dict, Iterable, List, Union

import numpy

from sparsezoo.utils.helpers import clean_path, create_dirs


__all__ = [
    "NDARRAY_KEY",
    "load_numpy",
    "save_numpy",
    "load_numpy_list",
    "NumpyArrayBatcher",
    "tensor_export",
    "tensors_export",
]

NDARRAY_KEY = "ndarray"
_LOGGER = logging.getLogger(__name__)


def _fix_loaded_numpy(array) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
    if not isinstance(array, numpy.ndarray):
        tmp_arrray = array
        array = OrderedDict()
        for key, val in tmp_arrray.items():
            array[key] = val

    return array


def load_numpy(file_path: str) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
    """
    Load a numpy file into either an ndarray or an OrderedDict representing what
    was in the npz file
    :param file_path: the file_path to load
    :return: the loaded values from the file
    """
    file_path = clean_path(file_path)
    array = numpy.load(file_path)

    return _fix_loaded_numpy(array)


def save_numpy(
    array: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name: str,
    npz: bool = True,
):
    """
    Save a numpy array or collection of numpy arrays to disk

    :param array: the array or collection of arrays to save
    :param export_dir: the directory to export the numpy file into
    :param name: the name of the file to export to (without extension)
    :param npz: True to save as an npz compressed file, False for standard npy.
        Note, npy can only be used for single numpy arrays
    :return: the saved path
    """
    create_dirs(export_dir)
    export_path = os.path.join(export_dir, f"{name}.{'npz' if npz else 'npy'}")

    if isinstance(array, numpy.ndarray) and npz:
        numpy.savez_compressed(export_path, array)
    elif isinstance(array, numpy.ndarray):
        numpy.save(export_path, array)
    elif isinstance(array, Dict) and npz:
        numpy.savez_compressed(export_path, **array)
    elif isinstance(array, Dict):
        raise ValueError("Dict can only be exported to an npz file")
    elif isinstance(array, Iterable) and npz:
        numpy.savez_compressed(export_path, *[val for val in array])
    elif isinstance(array, Iterable):
        raise ValueError("Iterable can only be exported to an npz file")
    else:
        raise ValueError(f"Unrecognized type given for array {array}")

    return export_path


def load_numpy_from_tar(
    path: str,
) -> List[Union[numpy.ndarray, Dict[str, numpy.ndarray]]]:
    """
    Load numpy data into a list from a tar file.
    All files contained in the tar are expected to be the numpy files.

    :param path: path to the tarfile to load the numpy data from
    :return: the list of loaded numpy data, either arrays or ordereddicts of arrays
    """
    tar = tarfile.open(path, "r")
    files = tar.getmembers()
    files = sorted([file.name for file in files])
    data = []

    for file in files:
        extracted = BytesIO()
        extracted.write(tar.extractfile(file).read())
        extracted.seek(0)
        array = numpy.load(extracted)
        data.append(_fix_loaded_numpy(array))

    return data


def load_numpy_list(
    data: Union[str, Iterable[Union[str, numpy.ndarray, Dict[str, numpy.ndarray]]]],
) -> List[Union[numpy.ndarray, Dict[str, numpy.ndarray]]]:
    """
    Load numpy data into a list

    :param data: the data to load, one of:
        [folder path, iterable of file paths, iterable of numpy arrays]
    :return: the list of loaded data items
    """
    loaded = []

    if isinstance(data, str):
        if os.path.isfile(data) and tarfile.is_tarfile(data):
            data = load_numpy_from_tar(data)
        elif os.path.isfile(data) and ".np" in data:
            # treat as a numpy file to load from
            data = [load_numpy(data)]
        else:
            # load from directory or glob
            glob_path = os.path.join(data, "*") if os.path.isdir(data) else data
            data = sorted(glob.glob(glob_path))

    for dat in data:
        if isinstance(dat, str):
            dat = load_numpy(dat)

        loaded.append(dat)

    return data


class NumpyArrayBatcher(object):
    """
    Batcher instance to handle taking in dictionaries of numpy arrays,
    appending multiple items to them to increase their batch size,
    and then stack them into a single batched numpy array for all keys in the dicts.
    """

    def __init__(self):
        self._items = OrderedDict()  # type: Dict[str, List[numpy.ndarray]]
        self._batch_index = None

    def __len__(self):
        if len(self._items) == 0:
            return 0

        return len(self._items[list(self._items.keys())[0]])

    def append(self, item: Union[numpy.ndarray, Dict[str, numpy.ndarray]]):
        """
        Append a new item into the current batch.
        All keys and shapes must match the current state.

        :param item: the item to add for batching
        """
        if len(self) < 1 and isinstance(item, numpy.ndarray):
            self._items[NDARRAY_KEY] = [item]
        elif len(self) < 1:
            for key, val in item.items():
                self._items[key] = [val]
        elif isinstance(item, numpy.ndarray):
            if self._batch_index is None:
                self._batch_index = {NDARRAY_KEY: 0}
            if NDARRAY_KEY not in self._items:
                raise ValueError(
                    "numpy ndarray passed for item, but prev_batch does not contain one"
                )

            if item.shape != self._items[NDARRAY_KEY][0].shape:
                self._batch_index[NDARRAY_KEY] = 1

            if item.shape != self._items[NDARRAY_KEY][0].shape and (
                item.shape[0] != self._items[NDARRAY_KEY][0].shape[0]
                or item.shape[2:] != self._items[NDARRAY_KEY][0].shape[2:]
            ):
                raise ValueError(
                    (
                        f"item of numpy ndarray of shape {item.shape} does not "
                        f"match the current batch shape of "
                        f"{self._items[NDARRAY_KEY][0].shape}"
                    )
                )

            self._items[NDARRAY_KEY].append(item)
        else:
            diff_keys = list(set(item.keys()) - set(self._items.keys()))

            if len(diff_keys) > 0:
                raise ValueError(
                    (
                        f"numpy dict passed for item, not all keys match "
                        f"with the prev_batch. difference: {diff_keys}"
                    )
                )

            if self._batch_index is None:
                self._batch_index = {key: 0 for key in item}

            for key, val in item.items():
                if val.shape != self._items[key][0].shape:
                    self._batch_index[key] = 1

                if val.shape != self._items[key][0].shape and (
                    val.shape[0] != self._items[key][0].shape[0]
                    or val.shape[2:] != self._items[key][0].shape[2:]
                ):
                    raise ValueError(
                        (
                            f"item with key {key} of shape {val.shape} does not "
                            f"match the current batch shape of "
                            f"{self._items[key][0].shape}"
                        )
                    )

                self._items[key].append(val)

    def stack(
        self, as_list: bool = False
    ) -> Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]:
        """
        Stack the current items into a batch along a new, zeroed dimension

        :param as_list: True to return the items as a list,
            False to return items in a named ordereddict
        :return: the stacked items
        """
        batch_dict = OrderedDict()

        for key, val in self._items.items():
            if self._batch_index is None or self._batch_index[key] == 0:
                batch_dict[key] = numpy.stack(val)
            else:
                batch_dict[key] = numpy.concatenate(val, axis=self._batch_index[key])
        return batch_dict if not as_list else list(batch_dict.values())


def tensor_export(
    tensor: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name: str,
    npz: bool = True,
) -> str:
    """
    :param tensor: tensor to export to a saved numpy array file
    :param export_dir: the directory to export the file in
    :param name: the name of the file, .npy will be appended to it
    :param npz: True to export as an npz file, False otherwise
    :return: the path of the numpy file the tensor was exported to
    """
    create_dirs(export_dir)
    export_path = os.path.join(export_dir, f"{name}.{'npz' if npz else 'npy'}")

    if isinstance(tensor, numpy.ndarray) and npz:
        numpy.savez_compressed(export_path, tensor)
    elif isinstance(tensor, numpy.ndarray):
        numpy.save(export_path, tensor)
    elif isinstance(tensor, Dict) and npz:
        numpy.savez_compressed(export_path, **tensor)
    elif isinstance(tensor, Dict):
        raise ValueError("tensor dictionaries can only be saved as npz")
    elif isinstance(tensor, Iterable) and npz:
        numpy.savez_compressed(export_path, *tensor)
    elif isinstance(tensor, Iterable):
        raise ValueError("tensor iterables can only be saved as npz")
    else:
        raise ValueError(f"unknown type give for tensor {tensor}")

    return export_path


def tensors_export(
    tensors: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name_prefix: str,
    counter: int = 0,
    break_batch: bool = False,
) -> List[str]:
    """
    :param tensors: the tensors to export to a saved numpy array file
    :param export_dir: the directory to export the files in
    :param name_prefix: the prefix name for the tensors to save as, will append
        info about the position of the tensor in a list or dict in addition
        to the .npy file format
    :param counter: the current counter to save the tensor at
    :param break_batch: treat the tensor as a batch and break apart into
        multiple tensors
    :return: the exported paths
    """
    create_dirs(export_dir)
    exported_paths = []

    if break_batch:
        _tensors_export_batch(tensors, export_dir, name_prefix, counter, exported_paths)
    else:
        _tensors_export_recursive(
            tensors, export_dir, name_prefix, counter, exported_paths
        )

    return exported_paths


def _tensors_export_recursive(
    tensors: Union[numpy.ndarray, Iterable[numpy.ndarray]],
    export_dir: str,
    name_prefix: str,
    counter: int,
    exported_paths: List[str],
):
    if isinstance(tensors, numpy.ndarray):
        exported_paths.append(
            tensor_export(tensors, export_dir, f"{name_prefix}-{counter:04d}")
        )

        return

    if isinstance(tensors, Dict):
        raise ValueError("tensors dictionary is not supported for non batch export")

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(tensors):
            _tensors_export_recursive(
                tens,
                export_dir,
                name_prefix,
                counter + index,
                exported_paths,
            )

        return

    raise ValueError(
        f"unrecognized type for tensors given of {tensors.__class__.__name__}"
    )


def _tensors_export_batch(
    tensors: Union[numpy.ndarray, Dict[str, numpy.ndarray], Iterable[numpy.ndarray]],
    export_dir: str,
    name_prefix: str,
    counter: int,
    exported_paths: List[str],
):
    if isinstance(tensors, numpy.ndarray):
        for index, tens in enumerate(tensors):
            exported_paths.append(
                tensor_export(tens, export_dir, f"{name_prefix}-{counter + index:04d}")
            )

        return

    if isinstance(tensors, Dict):
        tensors = OrderedDict([(key, val) for key, val in tensors.items()])
        keys = [key for key in tensors.keys()]

        for index, tens in enumerate(zip(*tensors.values())):
            tens = OrderedDict([(key, val) for key, val in zip(keys, tens)])
            exported_paths.append(
                tensor_export(tens, export_dir, f"{name_prefix}-{counter + index:04d}")
            )

        return

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(zip(*tensors)):
            exported_paths.append(
                tensor_export(tens, export_dir, f"{name_prefix}-{counter + index:04d}")
            )

        return

    raise ValueError(
        f"unrecognized type for tensors given of {tensors.__class__.__name__}"
    )
