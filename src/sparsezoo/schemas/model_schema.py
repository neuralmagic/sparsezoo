"""
Code related to a model repo model
"""

import os
import logging
from typing import Dict, List, Union

from sparsezoo.schemas.file_schema import RepoFile
from sparsezoo.schemas.tag_schema import RepoTag
from sparsezoo.schemas.optimization_schema import RepoOptimization
from sparsezoo.schemas.release_version_schema import RepoReleaseVersion
from sparsezoo.schemas.result_schema import RepoResult
from sparsezoo.schemas.user_schema import RepoUser
from sparsezoo.schemas.downloadable_schema import RepoDownloadable
from sparsezoo.utils import create_dirs

__all__ = ["RepoModel"]

_LOGGER = logging.getLogger(__name__)

FRAMEWORK_FILE_TYPE = "framework"
ONNX_FILE_TYPE = "onnx"
OPTIMIZATION_FILE_TYPE = "optimization"
DATA_FILE_TYPES = set(["inputs", "outputs", "labels"])


class RepoModel(RepoDownloadable):
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
    :param release_version_id: the release version id for the release version of the model
    :param base_model: the model id of a model this model inherited from
    :param files: a list of model repo files for this model
    :param tags: a list of model repo tags for this model
    :param user: the model repo user who uploaded this model
    :param optimizations: a list of model repo optimizations for this model
    :param results: a list of model repo results for this model
    :param release_version: a model repo release version this model was released with
    """

    def __init__(self, **kwargs):
        super(RepoModel, self).__init__(**kwargs)

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

        if "files" in kwargs:
            self._files = [RepoFile(**file) for file in kwargs["files"]]
        else:
            self._files = []

        if "tags" in kwargs:
            self._tags = [RepoTag(**tag) for tag in kwargs["tags"]]
        else:
            self._tags = []

        if "optimizations" in kwargs:
            self._optimizations = [
                RepoOptimization(**optimization)
                for optimization in kwargs["optimizations"]
            ]
        else:
            self._optimizations = []

        if "results" in kwargs:
            self._results = [RepoResult(**result) for result in kwargs["results"]]
        else:
            self._results = []

        if "release_version" in kwargs:
            self._release_version = RepoReleaseVersion(**kwargs["release_version"])
        else:
            self._release_version = None

        if "user" in kwargs:
            self._user = RepoUser(**kwargs["user"])
        else:
            self._user = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def sub_domain(self) -> str:
        return self._sub_domain

    @property
    def architecture(self) -> str:
        return self._architecture

    @property
    def sub_architecture(self) -> str:
        return self._sub_architecture

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def framework(self) -> str:
        return self._framework

    @property
    def optimization_name(self) -> str:
        return self._optimization_name

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def display_description(self) -> str:
        return self._display_description

    @property
    def repo_source(self) -> str:
        return self._repo_source

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def release_version_id(self) -> str:
        return self._release_version_id

    @property
    def base_model(self) -> str:
        return self._base_model

    @property
    def files(self) -> List[RepoFile]:
        return self._files

    @property
    def tags(self) -> List[RepoTag]:
        return self._tags

    @property
    def optimizations(self) -> List[RepoOptimization]:
        return self._optimizations

    @property
    def results(self) -> List[RepoResult]:
        return self._results

    @property
    def release_version(self) -> Union[RepoReleaseVersion, None]:
        return self._release_version

    @property
    def user(self) -> Union[RepoUser, None]:
        return self._user

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

    @property
    def framework_files(self) -> List[RepoFile]:
        return [
            file_obj
            for file_obj in self.files
            if file_obj.file_type == FRAMEWORK_FILE_TYPE
        ]

    @property
    def optimization_files(self) -> List[RepoFile]:
        return [
            file_obj
            for file_obj in self.files
            if file_obj.file_type == OPTIMIZATION_FILE_TYPE
        ]

    @property
    def onnx_files(self) -> List[RepoFile]:
        return [
            file_obj for file_obj in self.files if file_obj.file_type == ONNX_FILE_TYPE
        ]

    @property
    def data_files(self) -> List[RepoFile]:
        return [
            file_obj for file_obj in self.files if file_obj.file_type in DATA_FILE_TYPES
        ]

    def _download_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        files=List[RepoFile],
    ) -> str:
        save_folder = self._get_download_folder(overwrite, save_dir, save_path)

        for file_obj in files:
            file_obj.download(
                overwrite=overwrite,
                save_dir=save_folder,
            )
        return save_folder

    def download(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        """
        Downloads all files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id}.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.files,
        )

    def download_onnx_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        """
        Downloads all onnx files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} onnx files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.onnx_files,
        )

    def download_framework_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        """
        Downloads all framework files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} framework files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.framework_files,
        )

    def download_data_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        """
        Downloads all data files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} data files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.data_files,
        )

    def download_optimization_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
    ) -> str:
        """
        Downloads all optimization files associated with this model.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param save_dir: The directory to save the model files to
            instead of the default cache dir
        :param save_path: The exact path to save the model files to instead of
            the default cache dir or save_dir
        :return: the folder where the files were saved
        """
        _LOGGER.info(f"Downloading model {self.model_id} optimization files.")

        return self._download_files(
            overwrite=overwrite,
            save_dir=save_dir,
            save_path=save_path,
            files=self.optimization_files,
        )

    def dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "display_description": self.display_description,
            "domain": self.domain,
            "sub_domain": self.sub_domain,
            "architecture": self.architecture,
            "sub_architecture": self.sub_architecture,
            "dataset": self.dataset,
            "framework": self.framework,
            "optimization_name": self.optimization_name,
            "repo_source": self.repo_source,
            "base_model": self.base_model,
            "files": [file.dict() for file in self.files],
            "tags": [tag.dict() for tag in self.tags],
            "optimizations": [optim.dict() for optim in self.optimizations],
            "release_version": self.release_version.dict(),
            "results": [result.dict() for result in self.results],
            "user": self.user.dict(),
        }
