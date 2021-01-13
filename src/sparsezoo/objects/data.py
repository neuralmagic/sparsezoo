"""
Code related to a model repo optimization file
"""

import logging

from sparsezoo.utils import DataLoader
from sparsezoo.objects.file import File
from sparsezoo.objects.metadata import ModelMetadata


__all__ = ["Data"]

_LOGGER = logging.getLogger(__name__)


class Data(File):
    """
    A model repo optimization recipe.

    :param model_metadata: the metadata for the model the file is for
    :param recipe_id: the recipe id
    :param recipe_type: the type of optimization recipe
    :param display_name: the display name for the optimization
    :param display_description: the display description for the optimization
    """

    def __init__(
        self, model_metadata: ModelMetadata, **kwargs,
    ):
        super(Data, self).__init__(model_metadata=model_metadata, **kwargs)

    def loader(self, batch_size: int, iter_steps: int = 0) -> DataLoader:
        if not self.downloaded:
            raise RuntimeError(
                "data files must be downloaded first before creating a data loader"
            )

        files_dir = self.path.replace(".tar.gz", "")

        return DataLoader(files_dir, batch_size=batch_size, iter_steps=iter_steps)
