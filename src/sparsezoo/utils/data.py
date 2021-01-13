"""
Utilities for data loading into numpy for use in ONNX supported systems
"""

import logging
import math
from typing import Dict, List, Iterable, Tuple, Union

import numpy

from sparsezoo.utils.numpy import NumpyArrayBatcher, load_grouped_data


__all__ = ["DataLoader", "RandomDataLoader"]


_LOGGER = logging.getLogger(__name__)


class DataLoader(object):
    """
    Data loader instance that supports loading numpy arrays from file or memory
    and creating an iterator to go through batches of that data.
    Iterator returns a list containing all data originally loaded.

    :param args: any number of: file glob pointing to numpy files or loaded numpy data
    :param batch_size: the size of batches to create for the iterator
    :param iter_steps: the number of steps (batches) to create.
        Set to -1 for infinite, 0 for running through the loaded data once,
        or a positive integer for the desired number of steps
    """

    def __init__(
        self,
        *args: Union[str, List[Dict[str, numpy.ndarray]]],
        batch_size: int,
        iter_steps: int = 0,
    ):
        self._batch_size = batch_size
        self._iter_steps = iter_steps
        self._grouped_data = load_grouped_data(*args, raise_on_error=False)

        if len(self._grouped_data) < 1:
            raise ValueError(
                "No data for DataLoader after loading. data: {}".format(args)
            )

        self._index = 0
        self._step_count = 0

        if self.infinite:
            # __len__ cannot return math.inf as a value and must be non-negative integer
            self._max_steps = 0
        elif self._iter_steps > 0:
            self._max_steps = self._iter_steps
        else:
            self._max_steps = math.ceil(
                len(self._grouped_data) / float(self._batch_size)
            )

    @property
    def batch_size(self) -> int:
        """
        :return: the size of batches to create for the iterator
        """
        return self._batch_size

    @property
    def iter_steps(self) -> int:
        """
        :return: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        """
        return self._iter_steps

    @property
    def grouped_data(
        self,
    ) -> List[List[Union[numpy.ndarray, Dict[str, numpy.ndarray]]]]:
        """
        :return: the loaded data and labels
        """
        return self._grouped_data

    @property
    def infinite(self) -> bool:
        """
        :return: True if the loader instance is setup to continually create batches,
            False otherwise
        """
        return self._iter_steps == -1

    def __len__(self):
        return self._max_steps

    def __iter__(self):
        self._index = 0
        self._step_count = 0

        return self

    def __next__(self) -> List[Dict[str, numpy.ndarray]]:
        if not self.infinite and self._step_count >= self._max_steps:
            _LOGGER.debug("reached in of dataset, raising StopIteration")
            raise StopIteration()

        self._step_count += 1
        batchers = [NumpyArrayBatcher() for _ in self._grouped_data[0]]
        num_resets = 0

        while len(batchers[0]) < self._batch_size:
            try:
                _LOGGER.debug("including data in batch at index {}".format(self._index))
                data = self._grouped_data[self._index]

                for dat, batcher in zip(data, batchers):
                    batcher.append(dat)
            except Exception as err:
                logging.error(
                    (
                        "DataLoader: Error while adding file "
                        "to batch for index {}: {}"
                    ).format(self._index, err)
                )

            if self._index >= len(self._grouped_data) - 1:
                _LOGGER.debug("resetting index to loop data again")
                self._index = 0
                num_resets += 1

                if num_resets > self._batch_size // len(self._grouped_data) + 2:
                    # make sure we're not in an infinite loop because none of the
                    # data was loadable
                    raise ValueError(
                        "could not create a batch from the files, "
                        "not enough were loadable to fill the batch size"
                    )
            else:
                self._index += 1

        batched_data = [batcher.stack() for batcher in batchers]
        _LOGGER.debug("created batch data of size {}".format(len(batched_data[0])))

        return batched_data


class RandomDataLoader(DataLoader):
    """
    Data loader instance that supports creating random numpy arrays
    and creating an iterator to go through batches of that data.
    Iterator returns a list containing data for all shapes originally loaded.

    :param args: any number of dictionaries containing the key name mapped to
        a list of shapes, dtype
    :param batch_size: the size of batches to create for the iterator
    :param iter_steps: the number of steps (batches) to create.
        Set to -1 for infinite, 0 for running through the loaded data once,
        or a positive integer for the desired number of steps
    """

    def __init__(
        self,
        *args: Dict[str, Tuple[Iterable[int], Union[numpy.dtype, None]]],
        batch_size: int,
        iter_steps: int = 0,
        num_samples: int = 20,
    ):
        data_sets = []

        for data_desc in args:
            for (shape, type_) in data_desc.values():
                if not all(isinstance(dim, int) and dim > 0 for dim in shape):
                    raise RuntimeError(
                        "Invalid input shape, cannot create a random input shape"
                        " from: {}".format(shape)
                    )

            data = []
            for _ in range(num_samples):
                item = {}

                for key, (shape, type_) in data_desc.items():
                    dtype = type_ if type_ else numpy.float32
                    dtype_name = (
                        dtype.name if hasattr(dtype, "name") else dtype.__name__
                    )

                    if "float" in dtype_name:
                        array = numpy.random.random(shape).astype(dtype)
                    elif "int" in dtype_name:
                        iinfo = numpy.iinfo(dtype)
                        array = numpy.random.randint(iinfo.min, iinfo.max, shape, dtype)
                    elif dtype is numpy.bool:
                        array = numpy.random.random(shape) < 0.5
                    else:
                        raise RuntimeError(
                            "Cannot create random input for"
                            " {} with unsupported type {}".format(key, dtype)
                        )

                    item[key] = array
                data.append(item)
            data_sets.append(data)

        super(RandomDataLoader, self).__init__(
            *data_sets, batch_size=batch_size, iter_steps=iter_steps
        )
