"""
Code related to a model repo model
"""

import logging
import os
from typing import List, Union

import requests
from sparsezoo.models.downloadable import RepoDownloadable
from sparsezoo.models.file import File, UnsignedFileError
from sparsezoo.models.optimization import OptimizationRecipe
from sparsezoo.models.release_version import ReleaseVersion
from sparsezoo.models.result import Result
from sparsezoo.models.tag import Tag
from sparsezoo.models.user import User
from sparsezoo.utils import BASE_API_URL, create_dirs, get_auth_header


__all__ = ["Model"]

_LOGGER = logging.getLogger(__name__)

FRAMEWORK_FILE_TYPE = "framework"
ONNX_FILE_TYPE = "onnx"
OPTIMIZATION_FILE_TYPE = "optimization"
DATA_FILE_TYPES = set(["inputs", "outputs", "labels"])


class Model(RepoDownloadable):
    """
    A model repo model

    :param model_id: the model id
    :param domain: The domain of the models e.g. cv
    :param sub_domain: The sub domain of the models e.g. classification
    :param architecture: The architecture of the models e.g. mobilenet
    :param sub_architecture: The sub architecture of the model e.g. 1.0
    :param dataset: The dataset the model was trained on e.g. imagenet
    :param framework: The framework the model was trained on e.g. pytorch
    :param optimization_name: The level of optimization of the model e.g. base
    :param display_name: the display name for the model
    :param display_description: the description for the model
    :param repo_source: the source repo for the model
    :param user_id: the user id for the user who uploaded model
    :param release_version_id: the release version id for the release version of the
        model
    :param base_model: the model id of a model this model inherited from
    :param files: a list of model repo files for this model
    :param tags: a list of model repo tags for this model
    :param user: the model repo user who uploaded this model
    :param optimizations: a list of model repo optimizations for this model
    :param results: a list of model repo results for this model
    :param release_version: a model repo release version this model was released with
    :param tag_line: the tag line for the model
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        self._model_id = kwargs["model_id"]
        self._domain = kwargs["domain"]
        self._sub_domain = kwargs["sub_domain"]
        self._architecture = kwargs["architecture"]
        self._sub_architecture = kwargs["sub_architecture"]
        self._dataset = kwargs["dataset"]
        self._framework = kwargs["framework"]
        self._optimization_name = kwargs["optimization_name"]
        self._display_name = kwargs["display_name"]
        self._display_description = kwargs["display_description"]
        self._repo_source = kwargs["repo_source"]

        self._user_id = kwargs["user_id"]
        self._release_version_id = kwargs["release_version_id"]
        self._base_model = kwargs["base_model"]
        self._tag_line = kwargs["tag_line"]

        if "files" in kwargs:
            self._files = [File(**file) for file in kwargs["files"]]
        else:
            self._files = []

        if "tags" in kwargs:
            self._tags = [Tag(**tag) for tag in kwargs["tags"]]
        else:
            self._tags = []

        if "optimizations" in kwargs:
            self._optimizations = [
                OptimizationRecipe(**optimization)
                for optimization in kwargs["optimizations"]
            ]
        else:
            self._optimizations = []

        if "results" in kwargs:
            self._results = [Result(**result) for result in kwargs["results"]]
        else:
            self._results = []

        if "release_version" in kwargs:
            self._release_version = ReleaseVersion(**kwargs["release_version"])
        else:
            self._release_version = None

        if "user" in kwargs:
            self._user = User(**kwargs["user"])
        else:
            self._user = None

    @staticmethod
    def get_downloadable_model(
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: str,
        dataset: str,
        framework: str,
        optimization_name: str,
        release_version: str = None,
        force_token_refresh: bool = False,
    ):
        """
        Obtains a Model with signed files from the model repo

        :param domain: The domain of the models e.g. cv
        :param sub_domain: The sub domain of the models e.g. classification
        :param architecture: The architecture of the models e.g. mobilenet
        :param sub_architecture: The sub architecture of the model e.g. 1.0
        :param dataset: The dataset the model was trained on e.g. imagenet
        :param framework: The framework the model was trained on e.g. pytorch
        :param optimization_name: The level of optimization of the model e.g. base
        :param release_version: Optional param specifying the maximum supported
            release version for the models
        :param force_token_refresh: Forces a refresh of the authentication token
        :return: the Model
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
        )
        if release_version:
            url += f"?release_version={release_version}"
        _LOGGER.info(f"Obtaining model from {url}")

        response = requests.get(url=url, headers=header)
        response.raise_for_status()
        response_json = response.json()

        return Model(**response_json["model"])

    @property
    def model_id(self) -> str:
        """
        :return: the model id
        """
        return self._model_id

    @property
    def domain(self) -> str:
        """
        :return: The domain of the models e.g. cv
        """
        return self._domain

    @property
    def sub_domain(self) -> str:
        """
        :return: The sub domain of the models e.g. classification
        """
        return self._sub_domain

    @property
    def architecture(self) -> str:
        """
        :return: The architecture of the models e.g. mobilenet
        """
        return self._architecture

    @property
    def sub_architecture(self) -> str:
        """
        :return: The sub architecture of the model e.g. 1.0
        """
        return self._sub_architecture

    @property
    def dataset(self) -> str:
        """
        :return: The dataset the model was trained on e.g. imagenet
        """
        return self._dataset

    @property
    def framework(self) -> str:
        """
        :return: The framework the model was trained on e.g. pytorch
        """
        return self._framework

    @property
    def optimization_name(self) -> str:
        """
        :return: The level of optimization of the model e.g. base
        """
        return self._optimization_name

    @property
    def display_name(self) -> str:
        """
        :return: the display name for the model
        """
        return self._display_name

    @property
    def display_description(self) -> str:
        """
        :return: the description for the model
        """
        return self._display_description

    @property
    def repo_source(self) -> str:
        """
        :return: the source repo for the model
        """
        return self._repo_source

    @property
    def user_id(self) -> str:
        """
        :return: the user id for the user who uploaded model
        """
        return self._user_id

    @property
    def release_version_id(self) -> str:
        """
        :return: the release version id for the release version of the model
        """
        return self._release_version_id

    @property
    def base_model(self) -> str:
        """
        :return: the model id of a model this model inherited from
        """
        return self._base_model

    @property
    def files(self) -> List[File]:
        """
        :return: a list of model repo files for this model
        """
        return self._files

    @property
    def tags(self) -> List[Tag]:
        """
        :return: a list of model repo tags for this model
        """
        return self._tags

    @property
    def optimizations(self) -> List[OptimizationRecipe]:
        """
        :return: a list of model repo optimizations for this model
        """
        return self._optimizations

    @property
    def results(self) -> List[Result]:
        """
        :return: a list of model repo results for this model
        """
        return self._results

    @property
    def release_version(self) -> Union[ReleaseVersion, None]:
        """
        :return: a model repo release version this model was released with
        """
        return self._release_version

    @property
    def user(self) -> Union[User, None]:
        """
        :return: the model repo user who uploaded this model
        """
        return self._user

    @property
    def framework_files(self) -> List[File]:
        """
        :return: list of Files that are of type framework
        """
        return [
            file_obj
            for file_obj in self.files
            if file_obj.file_type == FRAMEWORK_FILE_TYPE
        ]

    @property
    def optimization_files(self) -> List[File]:
        """
        :return: list of Files that are of type optimization
        """
        return [
            file_obj
            for file_obj in self.files
            if file_obj.file_type == OPTIMIZATION_FILE_TYPE
        ]

    @property
    def onnx_files(self) -> List[File]:
        """
        :return: list of Files that are of type onnx
        """
        return [
            file_obj for file_obj in self.files if file_obj.file_type == ONNX_FILE_TYPE
        ]

    @property
    def tag_line(self) -> str:
        """
        :return: the tag line for model
        """
        return self._tag_line

    @property
    def data_files(self) -> List[File]:
        """
        :return: list of Files that are of type inputs, outputs, or labels
        """
        return [
            file_obj for file_obj in self.files if file_obj.file_type in DATA_FILE_TYPES
        ]

    def download(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        force_download_on_unsigned: bool = False,
    ) -> str:
        """
        Downloads all files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :param force_download_on_unsigned: If files are unsigned, updates all
            the model to contain signed version of files
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id}.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.files,
            force_download_on_unsigned=force_download_on_unsigned,
        )

    def download_onnx_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        force_download_on_unsigned: bool = False,
    ) -> str:
        """
        Downloads all onnx files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :param force_download_on_unsigned: If files are unsigned, updates all the model
            to contain signed version of files
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} onnx files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.onnx_files,
            force_download_on_unsigned=force_download_on_unsigned,
        )

    def download_framework_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        force_download_on_unsigned: bool = False,
    ) -> str:
        """
        Downloads all framework files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :param force_download_on_unsigned: If files are unsigned, updates all the model
            to contain signed version of files
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} framework files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.framework_files,
            force_download_on_unsigned=force_download_on_unsigned,
        )

    def download_data_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        force_download_on_unsigned: bool = False,
    ) -> str:
        """
        Downloads all data files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :param force_download_on_unsigned: If files are unsigned, updates all the model
            to contain signed version of files
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} data files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.data_files,
            force_download_on_unsigned=force_download_on_unsigned,
        )

    def download_optimization_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        force_download_on_unsigned: bool = False,
    ) -> str:
        """
        Downloads all optimization files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :param force_download_on_unsigned: If files are unsigned, updates all the model
            to contain signed version of files
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} optimization files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.optimization_files,
            force_download_on_unsigned=force_download_on_unsigned,
        )

    def _get_download_folder(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ):
        if save_dir:
            save_dir = os.path.join(save_dir, self.model_id)
        save_folder = self._get_download_save_path(overwrite, save_dir, save_path)

        if not os.path.exists(save_folder):
            create_dirs(save_folder)
        return save_folder

    def _download_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        files=List[File],
        force_download_on_unsigned: bool = False,
    ) -> str:
        save_folder = self._get_download_folder(overwrite, save_dir, save_path)
        for file_obj in files:
            try:
                file_obj.download(
                    overwrite=overwrite,
                    save_dir=save_folder,
                )
            except UnsignedFileError:
                if force_download_on_unsigned:
                    other_file = File.get_downloadable_file(
                        self.domain,
                        self.sub_domain,
                        self.architecture,
                        self.sub_architecture,
                        self.dataset,
                        self.framework,
                        self.optimization_name,
                        file_obj.display_name,
                        str(self.release_version),
                        force_token_refresh=True,
                    )
                    file_obj.url = other_file.url
                    file_obj.download(
                        overwrite=overwrite,
                        save_dir=save_folder,
                    )
                else:
                    raise
        return save_folder
