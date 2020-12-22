import os
import logging
from typing import NamedTuple, Union

from sparsezoo.schemas.object_schema import ObjectSchema
from sparsezoo.utils import clean_path, create_dirs

__all__ = ["DownloadableSchema", "ModelDownloadableSchema"]

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
    def __init__(self, default_save_path: str, default_file_name: str = "", **kwargs):
        super(DownloadableSchema, self).__init__(**kwargs)
        self._default_save_path = default_save_path
        self._default_file_name = default_file_name

    @property
    def default_save_path(self):
        return self._default_save_path

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
                    save_path = os.path.join("~", ".cache", "nm_models")

                save_dir = os.path.join(save_path, self._default_save_path)

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


class ModelDownloadableSchema(DownloadableSchema):
    def __init__(self, default_file_name: str = "", **kwargs):
        super(ModelDownloadableSchema, self).__init__(
            default_save_path=kwargs["model_id"],
            default_file_name=default_file_name,
            **kwargs
        )
        self._model_id = kwargs["model_id"]
