from enum import Enum
import logging
import os
import shutil
import tarfile

from sparsezoo.utils import (
    download_file,
    create_parent_dirs,
)
from sparsezoo.requests import download_get_request
from sparsezoo.objects.base import BaseObject
from sparsezoo.objects.downloadable import Downloadable
from sparsezoo.objects.metadata import ModelMetadata


__all__ = ["FileTypes", "File"]

_LOGGER = logging.getLogger(__name__)


class FileTypes(Enum):
    CARD = "card"
    ONNX = "onnx"
    RECIPE = "recipe"
    FRAMEWORK = "framework"
    DATA_ORIGINALS = "originals"
    DATA_INPUTS = "inputs"
    DATA_OUTPUTS = "outputs"
    DATA_LABELS = "labels"


class UnsignedFileError(Exception):
    """
    Error raised when a File does not contain signed url
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class File(BaseObject, Downloadable):
    """
    A model repo file.

    :param model_metadata: the metadata for the model the file is for
    :param file_id: the file id
    :param display_name: the display name for the optimization
    :param file_type: the file type e.g. onnx
    :param operator_version: Any operator version associated for the file
        e.g. onnx opset. None if no operator version exists
    :param checkpoint: True if this is a checkpoint file for transfer learning,
        False otherwise
    :param md5: The md5 hash of the file
    :param file_size: The file size in bytes
    :param downloads: The amount of times a download has been requested for
        this file
    :param url: The signed url to retrieve the file.
    """

    def __init__(
        self,
        model_metadata: ModelMetadata,
        file_id: str,
        display_name: str,
        file_type: str,
        operator_version: str,
        checkpoint: bool,
        md5: str,
        file_size: int,
        downloads: int,
        url: str = None,
        **kwargs,
    ):
        super(File, self).__init__(
            default_folder_name=model_metadata.model_id, **kwargs
        )
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
        return self._model_metadata

    @property
    def file_id(self) -> str:
        """
        :return: the file id
        """
        return self._file_id

    @property
    def display_name(self) -> str:
        """
        :return: the display name for the optimization
        """
        return self._display_name

    @property
    def file_type(self) -> str:
        """
        :return: the file type e.g. onnx
        """
        return self._file_type

    @property
    def file_type_card(self) -> bool:
        return self.file_type == FileTypes.CARD.value

    @property
    def file_type_onnx(self) -> bool:
        return self.file_type == FileTypes.ONNX.value

    @property
    def file_type_recipe(self) -> bool:
        return self.file_type == FileTypes.RECIPE.value

    @property
    def file_type_framework(self) -> bool:
        return self.file_type == FileTypes.FRAMEWORK.value

    @property
    def file_type_data(self) -> bool:
        return (
            self.file_type_data_originals
            or self.file_type_data_inputs
            or self.file_type_data_outputs
            or self.file_type_data_labels
        )

    @property
    def file_type_data_originals(self) -> bool:
        return self.file_type == FileTypes.DATA_ORIGINALS.value

    @property
    def file_type_data_inputs(self) -> bool:
        return self.file_type == FileTypes.DATA_INPUTS.value

    @property
    def file_type_data_outputs(self) -> bool:
        return self.file_type == FileTypes.DATA_OUTPUTS.value

    @property
    def file_type_data_labels(self) -> bool:
        return self.file_type == FileTypes.DATA_LABELS.value

    @property
    def operator_version(self) -> str:
        """
        :return: Any operator version associated for the file e.g. onnx opset.
            None if no operator version exists
        """
        return self._operator_version

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

    @property
    def path(self) -> str:
        return f"{self.dir_path}/{self.display_name}"

    @property
    def downloaded(self):
        return os.path.exists(self.path)

    def download(
        self,
        overwrite: bool = False,
        refresh_token: bool = False,
        show_progress: bool = True,
    ):
        """
        Downloads a model repo file. Will fail if the file does not contain a
        signed url. If file_type is either 'inputs', 'outputs', or 'labels',
        downloaded tar file will be extracted

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param refresh_token: refresh the auth token
        :param show_progress: True to use tqdm for progress, False to not show
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
        except:
            pass
        try:
            shutil.rmtree(self.path.replace(".tar.gz", ""))
        except:
            pass

        # creating target and downloading
        create_parent_dirs(self.path)
        download_file(
            self.url, self.path, overwrite=overwrite, show_progress=show_progress
        )

        if self.file_type_data:
            _LOGGER.info(f"extracting data tarfile at {self.path}")

            with tarfile.open(self.path) as tar:
                save_dir = os.path.dirname(self.path)
                tar.extractall(save_dir)

    def _signed_url(self, refresh_token: bool = False,) -> str:
        response_json = download_get_request(
            domain=self.model_metadata.domain,
            sub_domain=self.model_metadata.sub_domain,
            architecture=self.model_metadata.architecture,
            sub_architecture=self.model_metadata.sub_architecture,
            framework=self.model_metadata.framework,
            repo=self.model_metadata.repo,
            dataset=self.model_metadata.dataset,
            training_scheme=self.model_metadata.training_scheme,
            optim_name=self.model_metadata.optim_name,
            optim_category=self.model_metadata.optim_category,
            optim_target=self.model_metadata.optim_target,
            file_name=self.display_name,
            release_version=(
                str(self.model_metadata.release_version)
                if self.model_metadata.release_version
                else None
            ),
            force_token_refresh=refresh_token,
        )

        return response_json["file"]["url"]
