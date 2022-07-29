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
Utilities for data loading into numpy for use in ONNX supported systems
"""

import logging
import math
from collections import OrderedDict
from typing import Dict, Iterable, Iterator, List, Tuple, Union

import numpy

from sparsezoo.utils.numpy import NumpyArrayBatcher, load_numpy_list


__all__ = ["Dataset", "RandomDataset", "DataLoader"]

_LOGGER = logging.getLogger(__name__)


class Dataset(Iterable):
    """
    A numpy dataset implementation

    :param name: The name for the dataset
    :param data: The data for the dataset. Can be one of
        [str - path to a folder containing numpy files,
        Iterable[str] - list of paths to numpy files,
        Iterable[ndarray], Iterable[Dict[str, ndarray]]
        ]
    """

    def __init__(
        self,
        name: str,
        data: Union[str, Iterable[Union[str, numpy.ndarray, Dict[str, numpy.ndarray]]]],
    ):
        self._name = name
        self._data = load_numpy_list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index) -> Union[numpy.ndarray, Dict[str, numpy.ndarray]]:
        return self._data[index]

    def __iter__(self) -> Iterator[Union[numpy.ndarray, Dict[str, numpy.ndarray]]]:
        for item in self._data:
            yield item

    @property
    def name(self) -> str:
        """
        :return: The name for the dataset
        """
        return self._name

    @property
    def data(self) -> List[Union[numpy.ndarray, Dict[str, numpy.ndarray]]]:
        """
        :return: The list of data items for the dataset.
        """
        return self._data


class RandomDataset(Dataset):
    """
    A numpy dataset created from random data

    :param name: The name for the dataset
    :param typed_shapes: A dictionary containing the info for the random data to create,
        the names of the items in the data map to a tuple (shapes, numpy type).
        If numpy type is none, it will default to float32.
        Ex: {"inp": ([3, 224, 224], None)}
    :param num_samples: The number of random samples to create
    """

    def __init__(
        self,
        name: str,
        typed_shapes: Dict[str, Tuple[Iterable[int], Union[numpy.dtype, None]]],
        num_samples: int = 20,
    ):
        """

        :param name:
        :param typed_shapes:
        :param num_samples:
        """
        data = []
        for _ in range(num_samples):
            item = {}
            for key, (shape, type_) in typed_shapes.items():
                dtype = type_ if type_ else numpy.float32
                dtype_name = dtype.name if hasattr(dtype, "name") else dtype.__name__

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
                        f" {key} with unsupported type {dtype}"
                    )

                item[key] = array
            data.append(item)

        super().__init__(name, data)


class DataLoader(Iterable):
    """
    Data loader instance that supports loading numpy arrays from file or memory
    and creating an iterator to go through batches of that data.
    Iterator returns a list containing all data originally loaded.

    :param datasets: any number of datasets to load for the dataloader
    :param batch_size: the size of batches to create for the iterator
    :param iter_steps: the number of steps (batches) to create.
        Set to -1 for infinite, 0 for running through the loaded data once,
        or a positive integer for the desired number of steps
    :param batch_as_list: True to create the items from each dataset
        as a list, False for an ordereddict
    """

    def __init__(
        self,
        *datasets: Dataset,
        batch_size: int,
        iter_steps: int = 0,
        batch_as_list: bool = False,
    ):
        if len(datasets) < 1:
            raise ValueError("len(datasets) must be > 0")

        if batch_size < 1:
            raise ValueError("batch_size must be > 0")

        if iter_steps < -1:
            raise ValueError("iter_steps must be >= -1")

        self._datasets = OrderedDict([(dataset.name, dataset) for dataset in datasets])
        self._batch_size = batch_size
        self._iter_steps = iter_steps
        self._batch_as_list = batch_as_list
        self._num_items = -1

        for dataset in datasets:
            num_dataset_items = len(dataset)

            if num_dataset_items < 1:
                raise ValueError(f"No data in dataset: {dataset.name}")

            if self._num_items != -1 and num_dataset_items != self._num_items:
                raise ValueError(
                    f"Dataset {dataset.name} with length "
                    f"{num_dataset_items} does not match "
                    f"the other datasets of length {self._num_items}"
                )

            self._num_items = num_dataset_items

        self._iter_index = 0
        self._iter_step_count = 0

        if self.infinite:
            self._max_steps = 0
        elif self._iter_steps > 0:
            self._max_steps = self._iter_steps
        else:
            self._max_steps = math.ceil(self._num_items / float(self._batch_size))

    def __len__(self):
        return self._max_steps

    def __iter__(self):
        self._iter_index = 0
        self._iter_step_count = 0

        return self

    def __next__(
        self,
    ) -> Union[
        Dict[str, Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]],
        List[numpy.ndarray],
        Dict[str, numpy.ndarray],
    ]:
        if not self.infinite and self._iter_step_count >= self._max_steps:
            _LOGGER.debug("reached end of dataset, raising StopIteration")
            raise StopIteration()

        self._iter_step_count += 1
        end_index, batch = self._create_batch(self._iter_index)
        self._iter_index = end_index

        return batch

    @property
    def datasets(self) -> Dict[str, Dataset]:
        """
        :return: any number of datasets to load for the dataloader
        """
        return self._datasets

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
    def infinite(self) -> bool:
        """
        :return: True if the loader instance is setup to continually create batches,
            False otherwise
        """
        return self._iter_steps == -1

    @property
    def batch_as_list(self) -> bool:
        """
        :return: True to create the items from each dataset
            as a list, False for an ordereddict
        """
        return self._batch_as_list

    @property
    def num_items(self) -> int:
        """
        :return: the number of items in each dataset
        """
        return self._num_items

    def get_batch(
        self, bath_index: int
    ) -> Union[
        Dict[str, Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]],
        List[numpy.ndarray],
        Dict[str, numpy.ndarray],
    ]:
        """
        Get a batch from the data at the given index

        :param bath_index: the index of the batch to get
        :return: the created batch
        """
        max_batches = math.ceil(self._num_items / self._batch_size)

        if bath_index >= max_batches and not self.infinite:
            raise IndexError(
                f"batch_index {bath_index} is greater than the max {max_batches}"
            )

        start_index = bath_index * self._num_items
        _, batch = self._create_batch(start_index)

        return batch

    def _create_batch(
        self, start_index
    ) -> Tuple[
        int,
        Union[
            Dict[str, Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]],
            List[numpy.ndarray],
            Dict[str, numpy.ndarray],
        ],
    ]:
        _LOGGER.debug(f"creating batch at data start index {start_index}")
        dataset_batchers = [
            (dataset, NumpyArrayBatcher()) for key, dataset in self._datasets.items()
        ]
        index = start_index
        num_resets = 0
        num_items = len(dataset_batchers[0][0])
        max_resets = self._batch_size // num_items + 2

        if index >= self._num_items:
            index = 0

        while len(dataset_batchers[0][1]) < self._batch_size:
            try:
                _LOGGER.debug(f"including data in batch at index {index}")

                for dataset, batcher in dataset_batchers:
                    batcher.append(dataset[index])
            except Exception as err:
                raise RuntimeError(
                    f"DataLoader: Error while adding data "
                    f"to batch for index {index}: {err}"
                )

            index += 1

            if index >= num_items - 1:
                _LOGGER.debug("resetting index to loop data again")
                index = 0
                num_resets += 1

                if num_resets > max_resets:
                    # make sure we're not in an infinite loop because none of the
                    # data was loadable
                    raise ValueError(
                        "could not create a batch from the files, "
                        "not enough were loadable to fill the batch size"
                    )

        if len(dataset_batchers) == 1:
            # special case where only have a single dataset
            # for simplicity return the created batch instead nesting in dict
            return index, dataset_batchers[0][1].stack(as_list=self._batch_as_list)

        return (
            index,
            OrderedDict(
                [
                    (dataset.name, batcher.stack(as_list=self._batch_as_list))
                    for (dataset, batcher) in dataset_batchers
                ]
            ),
        )
