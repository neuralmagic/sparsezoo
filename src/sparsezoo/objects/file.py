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
Code related to files as stored for the sparsezoo as well a interacting with them
such as downloading
"""

import logging
import os
from enum import Enum
from typing import Union

from sparsezoo.objects.base import BaseObject
from sparsezoo.objects.downloadable import Downloadable
from sparsezoo.objects.metadata import ModelMetadata
from sparsezoo.requests import download_get_request
from sparsezoo.utils import create_parent_dirs, download_file


__all__ = ["FileTypes", "File"]

_LOGGER = logging.getLogger(__name__)


class FileTypes(Enum):
    """
    Types of files available in the sparsezoo
    """

    CARD = "card"
    ONNX = "onnx"
    ONNX_GZ = "onnx_gz"
    RECIPE = "recipe"
    FRAMEWORK = "framework"
    DATA_ORIGINALS = "originals"
    DATA_INPUTS = "inputs"
    DATA_OUTPUTS = "outputs"
    DATA_LABELS = "labels"


class File(BaseObject, Downloadable):
    """
    A model repo file.

    :param model_metadata: The metadata of the model the file belongs to
    :param file_id: Id of the file as stored in the cloud
    :param display_name: The file name and extension
    :param file_type: The type of file the object represents
    :param operator_version: Version of the file such as onnx OPSET for onnx files
    :param checkpoint: True if the model is a checkpoint file
        (for use with transfer learning flows), False otherwise
    :param md5: The md5 hash for the file as stored in the cloud
    :param file_size: The size of the file as stored in the cloud
    :param downloads: The amount of times a download has been requested for
        this file
    :param url: The signed url to retrieve the file
    :param child_folder_name: A child folder, if any, to store this file under locally
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
    """

    def __init__(
        self,
        model_metadata: ModelMetadata,
        file_id: str,
        display_name: str,
        file_type: str,
        operator_version: Union[str, None],
        checkpoint: bool,
        md5: str,
        file_size: int,
        downloads: int,
        url: str = None,
        child_folder_name: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        **kwargs,
    ):
        folder_name = (
            model_metadata.model_id
            if not override_folder_name
            else override_folder_name
        )
        if child_folder_name:
            folder_name = os.path.join(folder_name, child_folder_name)

        super(BaseObject, self).__init__(
            folder_name=folder_name,
            override_parent_path=override_parent_path,
            **kwargs,
        )
        super(File, self).__init__(**kwargs)
        self._model_metadata = model_metadata
        self._file_id = file_id
        self._display_name = display_name
        self._file_type = file_type
        self._operator_version = operator_version
        self._checkpoint = checkpoint
        self._md5 = md5
        self._file_size = file_size
        self._downloads = downloads
        self._url = url

    @property
    def model_metadata(self) -> ModelMetadata:
        """
        :return: The metadata of the model the file belongs to
        """
        return self._model_metadata

    @property
    def file_id(self) -> str:
        """
        :return: Id of the file as stored in the cloud
        """
        return self._file_id

    @property
    def display_name(self) -> str:
        """
        :return: The file name and extension
        """
        return self._display_name

    @property
    def file_type(self) -> str:
        """
        :return: The type of file the object represents
        """
        return self._file_type

    @property
    def file_type_card(self) -> bool:
        """
        :return: True if the file type is a card, False otherwise
        """
        return self.file_type == FileTypes.CARD.value

    @property
    def file_type_onnx(self) -> bool:
        """
        :return: True if the file type is onnx, False otherwise
        """
        return self.file_type == FileTypes.ONNX.value

    @property
    def file_type_onnx_gz(self) -> bool:
        """
        :return: True if the file type is a gzipped onnx, False otherwise
        """
        return self.file_type == FileTypes.ONNX_GZ.value

    @property
    def file_type_recipe(self) -> bool:
        """
        :return: True if the file type is a recipe, False otherwise
        """
        return self.file_type == FileTypes.RECIPE.value

    @property
    def file_type_framework(self) -> bool:
        """
        :return: True if the file type is a framework file, False otherwise
        """
        return self.file_type == FileTypes.FRAMEWORK.value

    @property
    def file_type_data(self) -> bool:
        """
        :return: True if the file type is sample data, False otherwise
        """
        return (
            self.file_type_data_originals
            or self.file_type_data_inputs
            or self.file_type_data_outputs
            or self.file_type_data_labels
        )

    @property
    def file_type_data_originals(self) -> bool:
        """
        :return: True if the file type is the original sample data, False otherwise
        """
        return self.file_type == FileTypes.DATA_ORIGINALS.value

    @property
    def file_type_data_inputs(self) -> bool:
        """
        :return: True if the file type is the input sample data, False otherwise
        """
        return self.file_type == FileTypes.DATA_INPUTS.value

    @property
    def file_type_data_outputs(self) -> bool:
        """
        :return: True if the file type is the output sample data, False otherwise
        """
        return self.file_type == FileTypes.DATA_OUTPUTS.value

    @property
    def file_type_data_labels(self) -> bool:
        """
        :return: True if the file type is the labels sample data, False otherwise
        """
        return self.file_type == FileTypes.DATA_LABELS.value

    @property
    def operator_version(self) -> Union[str, None]:
        """
        :return: Version of the file such as onnx OPSET for onnx files
        """
        return self._operator_version

    @property
    def checkpoint(self) -> bool:
        """
        :return: True if the model is a checkpoint file
            (for use with transfer learning flows), False otherwise
        """
        return self._checkpoint

    @property
    def md5(self) -> str:
        """
        :return: The md5 hash for the file as stored in the cloud
        """
        return self._md5

    @property
    def file_size(self) -> int:
        """
        :return: The size of the file as stored in the cloud
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

    @property
    def path(self) -> str:
        """
        :return: The path for where this file is (or can be) downloaded to
        """
        return f"{self.dir_path}/{self.display_name}"

    @property
    def downloaded(self):
        """
        :return: True if the file has already been downloaded, False otherwise
        """
        return os.path.exists(self.path)

    def downloaded_path(self) -> str:
        """
        :return: The local path to the downloaded file.
            Returns the same value as path, but if the file hasn't been downloaded
            then it will automatically download
        """
        self.check_download()

        return self.path

    def check_download(
        self,
        overwrite: bool = False,
        refresh_token: bool = False,
        show_progress: bool = True,
    ):
        """
        Check if the file has been downloaded, if not then call download()

        :param overwrite: True to overwrite any previous file, False otherwise
        :param refresh_token: True to refresh the auth token, False otherwise
        :param show_progress: True to print tqdm progress, False otherwise
        """
        if not self.downloaded or overwrite:
            self.download(overwrite, refresh_token, show_progress)

    def download(
        self,
        overwrite: bool = False,
        refresh_token: bool = False,
        show_progress: bool = True,
    ):
        """
        Downloads a sparsezoo file.

        :param overwrite: True to overwrite any previous file, False otherwise
        :param refresh_token: True to refresh the auth token, False otherwise
        :param show_progress: True to print tqdm progress, False otherwise
        """
        if os.path.exists(self.path) and not overwrite:
            _LOGGER.debug(
                f"Model file {self.display_name} already exists, "
                f"skipping download to {self.path}"
            )

            return

        if not self.url:
            _LOGGER.info(
                "Getting signed url for "
                f"{self.model_metadata.model_id}/{self.display_name}"
            )
            self._url = self._signed_url(refresh_token)

        _LOGGER.info(f"Downloading model file {self.display_name} to {self.path}")

        # cleaning up target
        try:
            os.remove(self.path)
        except Exception:
            pass

        # creating target and downloading
        create_parent_dirs(self.path)
        download_file(
            self.url, self.path, overwrite=overwrite, show_progress=show_progress
        )

    def _signed_url(
        self,
        refresh_token: bool = False,
    ) -> str:
        response_json = download_get_request(
            args=self.model_metadata,
            file_name=self.display_name,
            force_token_refresh=refresh_token,
        )

        return response_json["file"]["url"]
