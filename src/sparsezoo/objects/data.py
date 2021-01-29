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
Code related to sample data in the sparsezoo
"""

import logging
from typing import Dict, List, Union

import numpy

from sparsezoo.objects.file import File
from sparsezoo.objects.metadata import ModelMetadata
from sparsezoo.utils import DataLoader, Dataset


__all__ = ["Data"]

_LOGGER = logging.getLogger(__name__)


class Data(File):
    """
    Sample data for a given model

    :param name: The name/type of sample data
    :param model_metadata: Metadata for the model the data belongs to
    """

    def __init__(
        self,
        name: str,
        model_metadata: ModelMetadata,
        **kwargs,
    ):
        super(Data, self).__init__(model_metadata=model_metadata, **kwargs)
        self._name = name

    @property
    def name(self) -> str:
        """
        :return: The name/type of sample data
        """
        return self._name

    def dataset(self) -> Dataset:
        """
        A dataset for interacting with the sample data.
        If the data is not found on the local disk, will automatically download.

        :return: The created dataset from the sample data files
        """
        return Dataset(self._name, self.downloaded_path())

    def loader(
        self, batch_size: int = 1, iter_steps: int = 0, batch_as_list: bool = True
    ) -> DataLoader:
        """
        A dataloader for interfacing with the sample data in a batched format.

        :param batch_size: the size of the batches to create the loader for
        :param iter_steps: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        :param batch_as_list: True to return multiple inputs/outputs/etc
            within the dataset as lists, False for an ordereddict
        :return: The created dataloader from the sample data files
        """
        return DataLoader(
            self.dataset(),
            batch_size=batch_size,
            iter_steps=iter_steps,
            batch_as_list=batch_as_list,
        )

    def sample_batch(
        self, batch_index: int = 0, batch_size: int = 1, batch_as_list: bool = True
    ) -> Union[List[numpy.ndarray], Dict[str, numpy.ndarray]]:
        """
        Get a sample batch of data from the data loader

        :param batch_index: the index of the batch to get
        :param batch_size: the size of the batches to create the loader for
        :param batch_as_list: True to return multiple inputs/outputs/etc
            within the dataset as lists, False for an ordereddict
        :return: The sample batch for use with the model
        """
        loader = self.loader(batch_size=batch_size, batch_as_list=batch_as_list)

        return loader.get_batch(bath_index=batch_index)
