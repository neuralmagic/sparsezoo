from typing import Dict, Iterable, List, Union
from collections import OrderedDict
import glob
import os
import logging

import numpy

from sparsezoo.utils.helpers import clean_path, create_dirs


__all__ = [
    "NDARRAY_KEY",
    "load_numpy",
    "save_numpy",
    "load_grouped_data",
    "NumpyArrayBatcher",
    "tensor_export",
    "tensors_export",
]

NDARRAY_KEY = "ndarray"
_LOGGER = logging.getLogger(__name__)


def load_numpy(file_path: str) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
    """
    Load a numpy file into either an ndarray or an OrderedDict representing what
    was in the npz file
    :param file_path: the file_path to load
    :return: the loaded values from the file
    """
    file_path = clean_path(file_path)
    array = numpy.load(file_path)

    if not isinstance(array, numpy.ndarray):
        tmp_arrray = array
        array = OrderedDict()
        for key, val in tmp_arrray.items():
            array[key] = val

    return array


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
    export_path = os.path.join(
        export_dir, "{}.{}".format(name, "npz" if npz else "npy")
    )

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
        raise ValueError("Unrecognized type given for array {}".format(array))

    return export_path


def load_grouped_data(
    *args: Union[str, Iterable[Union[str, numpy.ndarray, Dict[str, numpy.ndarray]]]],
    raise_on_error: bool = True,
) -> List[List[Union[numpy.ndarray, Dict[str, numpy.ndarray]]]]:
    """
    Load data from disk or from memory and group them together.
    Assumes sorted ordering for on disk. Will match between when a file glob is passed
    for all data.

    :param args: any number of the file glob or list of arrays to use for data
    :param raise_on_error: True to raise on any error that occurs;
        False to log a warning, ignore, and continue
    :return: a list containing tuples of the data, labels. If labels was passed in
        as None, will now contain a None for the second index in each tuple
    """
    data_sets = []
    items_length = -1

    for data in args:
        if isinstance(data, str):
            data = sorted(glob.glob(data))
        data_sets.append(data)
        items_length = len(data)

    for data in data_sets:
        if len(data) != items_length:
            # always raise this error, lengths must match
            raise ValueError(
                "len(data) given of {} does not match with others at {}".format(
                    len(data), items_length
                )
            )

    grouped_data = []

    for data_set_index, data in enumerate(data_sets):
        for index, dat in enumerate(data):
            try:
                if isinstance(dat, str):
                    dat = load_numpy(dat)
            except Exception as err:
                if raise_on_error:
                    raise err
                else:
                    _LOGGER.error("Error grouping data: {}".format(err))

            if data_set_index == 0:
                grouped_data.append([])

            grouped_data[index].append(dat)

    return grouped_data


class NumpyArrayBatcher(object):
    """
    Batcher instance to handle taking in dictionaries of numpy arrays,
    appending multiple items to them to increase their batch size,
    and then stack them into a single batched numpy array for all keys in the dicts.
    """

    def __init__(self):
        self._items = OrderedDict()  # type: Dict[str, List[numpy.ndarray]]

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
            if NDARRAY_KEY not in self._items:
                raise ValueError(
                    "numpy ndarray passed for item, but prev_batch does not contain one"
                )

            if item.shape != self._items[NDARRAY_KEY][0].shape:
                raise ValueError(
                    (
                        "item of numpy ndarray of shape {} does not "
                        "match the current batch shape of {}".format(
                            item.shape, self._items[NDARRAY_KEY][0].shape
                        )
                    )
                )

            self._items[NDARRAY_KEY].append(item)
        else:
            diff_keys = list(set(item.keys()) - set(self._items.keys()))

            if len(diff_keys) > 0:
                raise ValueError(
                    (
                        "numpy dict passed for item, not all keys match "
                        "with the prev_batch. difference: {}"
                    ).format(diff_keys)
                )

            for key, val in item.items():
                if val.shape != self._items[key][0].shape:
                    raise ValueError(
                        (
                            "item with key {} of shape {} does not "
                            "match the current batch shape of {}".format(
                                key, val.shape, self._items[key][0].shape
                            )
                        )
                    )

                self._items[key].append(val)

    def stack(self) -> Dict[str, numpy.ndarray]:
        """
        Stack the current items into a batch along a new, zeroed dimension
        :return: the stacked items
        """
        batch_dict = OrderedDict()

        for key, val in self._items.items():
            batch_dict[key] = numpy.stack(self._items[key])

        return batch_dict


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
    export_path = os.path.join(
        export_dir, "{}.{}".format(name, "npz" if npz else "npy")
    )

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
        raise ValueError("unknown type give for tensor {}".format(tensor))

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
            tensor_export(tensors, export_dir, "{}-{:04d}".format(name_prefix, counter))
        )

        return

    if isinstance(tensors, Dict):
        raise ValueError("tensors dictionary is not supported for non batch export")

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(tensors):
            _tensors_export_recursive(
                tens, export_dir, name_prefix, counter + index, exported_paths,
            )

        return

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
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
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    if isinstance(tensors, Dict):
        tensors = OrderedDict([(key, val) for key, val in tensors.items()])
        keys = [key for key in tensors.keys()]

        for index, tens in enumerate(zip(*tensors.values())):
            tens = OrderedDict([(key, val) for key, val in zip(keys, tens)])
            exported_paths.append(
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    if isinstance(tensors, Iterable):
        for index, tens in enumerate(zip(*tensors)):
            exported_paths.append(
                tensor_export(
                    tens, export_dir, "{}-{:04d}".format(name_prefix, counter + index)
                )
            )

        return

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )
