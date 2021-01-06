"""
Code related to a model repo optimization file
"""
import logging

from sparsezoo.models.downloadable import RepoDownloadable
from sparsezoo.models.file import File


__all__ = ["OptimizationRecipe"]

_LOGGER = logging.getLogger(__name__)


class OptimizationRecipe(RepoDownloadable):
    """
    A model repo optimization.

    :param optimization_id: the optimization id
    :param model_id: the model id of the optimization
    :param display_name: the display name for the optimization
    :param display_description: the description for the optimization
    :param optimization_type: the type of optimizations
    :param tag_line: the tag line for optimization
    :param file: the file object for the optimization
    """

    def __init__(self, **kwargs):
        super(OptimizationRecipe, self).__init__(**kwargs)
        self._optimization_id = kwargs["optimization_id"]
        self._model_id = kwargs["model_id"]
        self._display_name = kwargs["display_name"]
        self._display_description = kwargs["display_description"]
        self._optimization_type = kwargs["optimization_type"]
        self._tag_line = kwargs["tag_line"]
        if "file" in kwargs:
            self._file = File(**kwargs["file"])
        else:
            self._file = None

    @property
    def display_name(self):
        """
        :return: the display name for the optimization
        """
        return self._display_name

    @property
    def display_description(self):
        """
        :return: the description for the optimization
        """
        return self._display_description

    @property
    def model_id(self) -> str:
        """
        :return: the model id of the optimization
        """
        return self._model_id

    @property
    def optimization_id(self) -> str:
        """
        :return: the optimization id
        """
        return self._optimization_id

    @property
    def tag_line(self) -> str:
        """
        :return: the tag line for optimization
        """
        return self._tag_line

    @property
    def optimization_type(self) -> str:
        """
        :return: the type of optimizations
        """
        return self._optimization_type

    @property
    def file(self) -> File:
        """
        :return: the file object for the optimization
        """
        return self._file

    def download(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        """
        Downloads the optimization yaml file associated with this optimization

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the optimization file to instead of the
            default cache dir
        :param save_path: The exact path to save the optimization file to instead of
            the default cache dir or save_dir
        :return: the folder where the file was saved
        """
        _LOGGER.info(
            f"Downloading recipe files for {self.display_name} "
            + "from model {self.model_id}."
        )
        if self.file is None:
            raise Exception(
                "No file found for optimization {}".format(self.display_name)
            )
        else:
            self.file.download(
                overwrite=overwrite, save_dir=save_dir, save_path=save_path
            )
