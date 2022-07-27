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

import glob
import os
import tarfile
from collections import OrderedDict
from io import BytesIO
from typing import Dict, Iterable, List, Union

import numpy

from .utils import clean_path, create_dirs


__all__ = ["save_numpy", "load_numpy_list"]


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

    return loaded
