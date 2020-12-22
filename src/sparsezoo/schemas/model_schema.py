import os
import logging
from typing import Dict, List, Union

from sparsezoo.schemas.file_schema import FileSchema
from sparsezoo.schemas.tag_schema import TagSchema
from sparsezoo.schemas.optimization_schema import OptimizationSchema
from sparsezoo.schemas.release_version_schema import ReleaseVersionSchema
from sparsezoo.schemas.result_schema import ResultSchema
from sparsezoo.schemas.user_schema import UserSchema
from sparsezoo.schemas.downloadable_schema import ModelDownloadableSchema
from sparsezoo.utils import create_dirs

__all__ = ["ModelSchema"]

_LOGGER = logging.getLogger(__name__)

FRAMEWORK_FILE_TYPE = "framework"
ONNX_FILE_TYPE = "onnx"
OPTIMIZATION_FILE_TYPE = "optimization"
DATA_FILE_TYPES = set(["inputs", "outputs", "labels"])


class ModelSchema(ModelDownloadableSchema):
    def __init__(self, **kwargs):
        super(ModelSchema, self).__init__(**kwargs)

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
            self._files = [FileSchema(**file) for file in kwargs["files"]]
        else:
            self._files = []

        if "tags" in kwargs:
            self._tags = [TagSchema(**tag) for tag in kwargs["tags"]]
        else:
            self._tags = []

        if "optimizations" in kwargs:
            self._optimizations = [
                OptimizationSchema(**optimization)
                for optimization in kwargs["optimizations"]
            ]
        else:
            self._optimizations = []

        if "results" in kwargs:
            self._results = [ResultSchema(**result) for result in kwargs["results"]]
        else:
            self._results = []

        if "release_version" in kwargs:
            self._release_version = ReleaseVersionSchema(**kwargs["release_version"])
        else:
            self._release_version = None

        if "user" in kwargs:
            self._user = UserSchema(**kwargs["user"])
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
    def files(self) -> List[FileSchema]:
        return self._files

    @property
    def tags(self) -> List[TagSchema]:
        return self._tags

    @property
    def optimizations(self) -> List[OptimizationSchema]:
        return self._optimizations

    @property
    def results(self) -> List[ResultSchema]:
        return self._results

    @property
    def release_version(self) -> Union[ReleaseVersionSchema, None]:
        return self._release_version

    @property
    def user(self) -> Union[UserSchema, None]:
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
    def framework_files(self) -> List[FileSchema]:
        return [
            file_obj
            for file_obj in self.files
            if file_obj.file_type == FRAMEWORK_FILE_TYPE
        ]

    @property
    def optimization_files(self) -> List[FileSchema]:
        return [
            file_obj
            for file_obj in self.files
            if file_obj.file_type == OPTIMIZATION_FILE_TYPE
        ]

    @property
    def onnx_files(self) -> List[FileSchema]:
        return [
            file_obj for file_obj in self.files if file_obj.file_type == ONNX_FILE_TYPE
        ]

    @property
    def data_files(self) -> List[FileSchema]:
        return [
            file_obj for file_obj in self.files if file_obj.file_type in DATA_FILE_TYPES
        ]

    def _download_files(
        self,
        overwrite: bool = False,
        save_dir: str = None,
        save_path: str = None,
        files=List[FileSchema],
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
