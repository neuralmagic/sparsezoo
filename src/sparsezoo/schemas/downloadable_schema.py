"""
Code related to a downloadable interface
"""

import os
import logging
from typing import NamedTuple, Union

from sparsezoo.schemas.object_schema import ObjectSchema
from sparsezoo.utils import clean_path, create_dirs, CACHE_DIR

__all__ = ["DownloadableSchema", "RepoDownloadable"]

_LOGGER = logging.getLogger(__name__)


DownloadProgress = NamedTuple(
    "DownloadProgress",
    [
        ("chunk_size", int),
        ("downloaded", int),
        ("content_length", Union[None, int]),
        ("path", str),
    ],
)


class DownloadableSchema(ObjectSchema):
    """
    Downloadable interface with a default folder and file name

    :param default_folder_name: Default folder to save file to save_dir or save_path are not provided
    :param default_file_name: Default file name to save file as if save_path is not provided
    """

    def __init__(
        self, default_folder_name: str = "", default_file_name: str = "", **kwargs
    ):
        super(DownloadableSchema, self).__init__(**kwargs)
        self._default_folder_name = default_folder_name
        self._default_file_name = default_file_name

    @property
    def default_folder_name(self):
        return self._default_folder_name

    @property
    def default_file_name(self):
        return self._default_file_name

    def _get_download_save_path(
        self, overwrite: bool = False, save_dir: str = None, save_path: str = None
    ):
        if save_path:
            save_file = clean_path(save_path)
            save_dir = os.path.dirname(save_path)
        else:
            if not save_dir:
                save_path = os.getenv("NM_ML_MODELS_PATH", "")

                if not save_path:
                    save_path = CACHE_DIR

                save_dir = os.path.join(save_path, self._default_folder_name)

            save_dir = clean_path(save_dir)
            save_file = os.path.join(save_dir, self.default_file_name)

        create_dirs(save_dir)

        if overwrite and os.path.exists(save_file):
            try:
                os.remove(save_file)
            except OSError as err:
                logging.warning(
                    "error encountered when removing older "
                    "cache_file at {}: {}".format(save_file, err)
                )
        return save_file

    def download(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ):
        raise NotImplementedError()


class RepoDownloadable(DownloadableSchema):
    """
    A downloadable model repo object. Uses model_id as the default folder to save the file

    :param default_folder_name: Default file name to save file as if save_path is not provided
    """

    def __init__(self, default_file_name: str = "", **kwargs):
        super(RepoDownloadable, self).__init__(
            default_folder_name=kwargs["model_id"],
            default_file_name=default_file_name,
            **kwargs,
        )
        self._model_id = kwargs["model_id"]
