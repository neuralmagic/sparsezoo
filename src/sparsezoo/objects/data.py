"""
Code related to sample data in the sparsezoo
"""

import logging

from sparsezoo.utils import DataLoader, Dataset
from sparsezoo.objects.file import File
from sparsezoo.objects.metadata import ModelMetadata


__all__ = ["Data"]

_LOGGER = logging.getLogger(__name__)


class Data(File):
    """
    Sample data for a given model

    :param name: The name/type of sample data
    :param model_metadata: Metadata for the model the data belongs to
    """

    def __init__(
        self, name: str, model_metadata: ModelMetadata, **kwargs,
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
        self, batch_size: int, iter_steps: int = 0, batch_as_list: bool = True
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
