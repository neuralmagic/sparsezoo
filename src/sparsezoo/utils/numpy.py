import numpy
from typing import Dict, Union, List, Iterable
from collections import OrderedDict
import tarfile
from io import BytesIO
import os
import glob

from .authentication import clean_path

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