"""
Code related to a downloadable interface
"""

from typing import Union
import logging
import os

from sparsezoo.utils import CACHE_DIR, clean_path, create_dirs


__all__ = ["Downloadable"]

_LOGGER = logging.getLogger(__name__)


class Downloadable:
    """
    Downloadable interface with a default folder and file name

    :param default_folder_name: Default folder to save file to save_dir or save_path are
        not provided
    """

    def __init__(
        self,
        default_folder_name: str,
        override_path: Union[str, None] = None,
        **kwargs,
    ):
        self._default_folder_name = default_folder_name
        self._override_path = override_path

    @property
    def default_folder_name(self):
        """
        :return: Default folder to save file to save_dir or save_path are not provided
        """
        return self._default_folder_name

    @property
    def dir_path(self) -> str:
        dir_path = self._override_path

        if not dir_path:
            dir_path = os.getenv("NM_ML_MODELS_PATH", "")

        if not dir_path:
            dir_path = CACHE_DIR

        dir_path = os.path.join(dir_path, self.default_folder_name)
        create_dirs(dir_path)

        return dir_path

    def download(
        self,
        overwrite: bool = False,
        refresh_token: bool = False,
        show_progress: bool = True,
    ):
        raise NotImplementedError()
