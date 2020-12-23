import os
import logging

from typing import Dict
from sparsezoo.schemas.downloadable_schema import RepoDownloadable
from sparsezoo.utils import download_file

__all__ = ["RepoFile"]

_LOGGER = logging.getLogger(__name__)


class RepoFile(RepoDownloadable):
    """
    A model repo file.

    :param file_id: the file id
    :param model_id: the model id of the optimization
    :param display_name: the display name for the optimization
    :param file_type: the file type e.g. onnx
    :param operator_version: Any operator version associated for the file e.g. onnx opset.
        None if no operator version exists
    :param optimization_id: An optimization id if this file is an optimization file, otherwise None
    :param quantized: True if file is a quantized onnx model. False otherwise.
    :param md5: The md5 hash of the file
    :param file_size: The file size in bytes
    :param downloads: The amount of times a download has been requested for this file
    :param url: The signed url to retrieve the file.
    """

    def __init__(self, **kwargs):
        super(RepoFile, self).__init__(kwargs["display_name"], **kwargs)
        self._file_id = kwargs["file_id"]
        self._model_id = kwargs["model_id"]
        self._display_name = kwargs["display_name"]
        self._file_type = kwargs["file_type"]
        self._operator_version = kwargs["operator_version"]
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
        """
        Downloads a model repo file. Will fail if the file does not contain a signed url

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the file to instead of the default cache dir
        :param save_path: The exact path to save the file to instead of the default cache dir or save_dir
        :return: the folder where the file was saved
        """
        if self.url is None:
            raise Exception(
                "File {} from model {} has not been signed.".format(
                    self.display_name, self.model_id
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
