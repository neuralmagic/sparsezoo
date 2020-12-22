import os
import logging

from typing import Dict
from sparsezoo.schemas.downloadable_schema import ModelDownloadableSchema
from sparsezoo.utils import download_file

__all__ = ["FileSchema"]

_LOGGER = logging.getLogger(__name__)


class FileSchema(ModelDownloadableSchema):
    def __init__(self, **kwargs):
        super(FileSchema, self).__init__(kwargs["display_name"], **kwargs)
        self._file_id = kwargs["file_id"]
        self._display_name = kwargs["display_name"]
        self._file_type = kwargs["file_type"]
        self._operator_version = kwargs["operator_version"]
        self._model_id = kwargs["model_id"]
        self._optimization_id = kwargs["optimization_id"]
        self._quantized = kwargs["quantized"]
        self._md5 = kwargs["md5"]
        self._file_size = kwargs["file_size"]
        self._downloads = kwargs["downloads"]
        if "url" in kwargs:
            self._url = kwargs["url"]
        else:
            self._url = None

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def file_id(self) -> str:
        return self._file_id

    @property
    def file_type(self) -> str:
        return self._file_type

    @property
    def operator_version(self) -> str:
        return self._operator_version

    @property
    def optimization_id(self) -> str:
        return self._optimization_id

    @property
    def quantized(self) -> str:
        return self._quantized

    @property
    def md5(self) -> str:
        return self._md5

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def downloads(self) -> int:
        return self._downloads

    @property
    def url(self) -> str:
        return self._url

    def download(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        if self.url is None:
            raise Exception(
                "File {} has not been signed. Please use download API to obtain a signed version of this model.".format(
                    self.model_id
                )
            )
        save_file = self._get_download_save_path(overwrite, save_dir, save_path)
        _LOGGER.info(f"Downloading model file {self.display_name} to {save_file}")

        if not os.path.exists(save_file) or overwrite:
            download_file(self.url, save_file, overwrite=overwrite)

        return save_file

    def dict(self) -> Dict:
        return {
            "display_name": self.display_name,
            "model_id": self.model_id,
            "file_id": self.file_id,
            "file_type": self.file_type,
            "operator_version": self.operator_version,
            "optimization_id": self.optimization_id,
            "quantized": self.quantized,
            "md5": self.md5,
            "file_size": self.file_size,
            "downloads": self.downloads,
            "url": self.url,
        }
