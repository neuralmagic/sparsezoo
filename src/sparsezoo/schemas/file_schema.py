import os
import logging

import requests

from sparsezoo.schemas.downloadable_schema import RepoDownloadable
from sparsezoo.utils import download_file, get_auth_header, BASE_API_URL

__all__ = ["File", "UnsignedFileError"]

_LOGGER = logging.getLogger(__name__)


class UnsignedFileError(Exception):
    """
    Error raised when a File does not contain signed url
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class File(RepoDownloadable):
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
        super(File, self).__init__(kwargs["display_name"], **kwargs)
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

    @staticmethod
    def get_downloadable_file(
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        dataset: str,
        framework: str,
        optimization_name: str,
        file_name: str,
        release_version: str = None,
        force_token_refresh: bool = False,
    ):
        """
        Obtains a File with a signed url from specified model attributes from the model repo.

        :param domain: The domain of the models e.g. cv
        :param sub_domain: The sub domain of the models e.g. classification
        :param architecture: The architecture of the models e.g. mobilenet
        :param sub_architecture: The sub architecture of the model e.g. 1.0
        :param dataset: The dataset the model was trained on e.g. imagenet
        :param framework: The framework the model was trained on e.g. pytorch
        :param optimization_name: The level of optimization of the model e.g. base
        :param file_name: The name of the file being downloaded e.g. model.onnx
        :param release_version: Optional param specifying the maximum supported release version for the models
        :param force_token_refresh: Forces a refresh of the authentication token
        :return: a File for the downloaded model file
        """
        header = get_auth_header(force_token_refresh=force_token_refresh)

        url = os.path.join(
            BASE_API_URL,
            "download",
            domain,
            sub_domain,
            architecture,
            sub_architecture,
            dataset,
            framework,
            optimization_name,
            file_name,
        )
        _LOGGER.info(f"Obtaining model file at {url}")

        if release_version:
            url += f"?release_version={release_version}"

        response = requests.get(url=url, headers=header)
        response.raise_for_status()
        response_json = response.json()

        return File(**response_json["file"])

    @property
    def display_name(self) -> str:
        """
        :return: the display name for the optimization
        """
        return self._display_name

    @property
    def model_id(self) -> str:
        """
        :return: the model id of the optimization
        """
        return self._model_id

    @property
    def file_id(self) -> str:
        """
        :return: the file id
        """
        return self._file_id

    @property
    def file_type(self) -> str:
        """
        :return: the file type e.g. onnx
        """
        return self._file_type

    @property
    def operator_version(self) -> str:
        """
        :return: Any operator version associated for the file e.g. onnx opset.
            None if no operator version exists
        """
        return self._operator_version

    @property
    def optimization_id(self) -> str:
        """
        :return: An optimization id if this file is an optimization file, otherwise None
        """
        return self._optimization_id

    @property
    def quantized(self) -> str:
        """
        :return: True if file is a quantized onnx model. False otherwise.
        """
        return self._quantized

    @property
    def md5(self) -> str:
        """
        :return: The md5 hash of the file
        """
        return self._md5

    @property
    def file_size(self) -> int:
        """
        :return: The file size in bytes
        """
        return self._file_size

    @property
    def downloads(self) -> int:
        """
        :return: The amount of times a download has been requested for this file
        """
        return self._downloads

    @property
    def url(self) -> str:
        """
        :return: The signed url to retrieve the file.
        """
        return self._url

    @url.setter
    def url(self, url):
        """
        Setter for url
        :param url: The signed url to retrieve the file.
        """
        self._url = url

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
            raise UnsignedFileError(
                "File {} from model {} has not been signed.".format(
                    self.display_name, self.model_id
                )
            )
        save_file = self._get_download_save_path(overwrite, save_dir, save_path)
        _LOGGER.info(f"Downloading model file {self.display_name} to {save_file}")

        if not os.path.exists(save_file) or overwrite:
            download_file(self.url, save_file, overwrite=overwrite)

        return save_file
