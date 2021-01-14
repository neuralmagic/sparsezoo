"""
Code related to a model from the sparsezoo
"""

from typing import List, Dict, Union, Any
import logging

from sparsezoo.requests import ModelArgs, download_get_request, search_get_request
from sparsezoo.utils import DataLoader
from sparsezoo.objects.downloadable import Downloadable
from sparsezoo.objects.metadata import ModelMetadata
from sparsezoo.objects.data import Data
from sparsezoo.objects.file import File, FileTypes
from sparsezoo.objects.optimization_recipe import OptimizationRecipe
from sparsezoo.objects.release_version import ReleaseVersion
from sparsezoo.objects.result import Result
from sparsezoo.objects.tag import Tag
from sparsezoo.objects.user import User


__all__ = ["Model"]

_LOGGER = logging.getLogger(__name__)


class Model(Downloadable, ModelMetadata):
    """
    A model repo model

    :param display_name: the display name for the model
    :param display_description: the description for the model
    :param files: a list of model repo files for this model
    :param recipes: a list of model repo optimization recipes for this model
    :param results: a list of model repo results for this model
    :param tags: a list of model repo tags for this model
    :param user: the model repo user who uploaded this model
    :param release_version: a model repo release version this model was released with
    :param tag_line: the tag line for the model
    """

    @staticmethod
    def get_downloadable(
        domain: str,
        sub_domain: str,
        architecture: str,
        sub_architecture: Union[str, None],
        framework: str,
        repo: str,
        dataset: str,
        training_scheme: Union[str, None],
        optim_name: str,
        optim_category: str,
        optim_target: Union[str, None],
        release_version: Union[str, None] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ):
        """
        Obtains a Model with signed files from the model repo

        [TODO]
        """
        args = ModelArgs(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
        )
        response_json = download_get_request(
            args=args, file_name=None, force_token_refresh=force_token_refresh,
        )

        return Model(
            **response_json["model"],
            override_folder_name=override_folder_name,
            override_parent_path=override_parent_path,
        )

    @staticmethod
    def search_downloadable(
        domain: str,
        sub_domain: str,
        architecture: Union[str, None] = None,
        sub_architecture: Union[str, None] = None,
        framework: Union[str, None] = None,
        repo: Union[str, None] = None,
        dataset: Union[str, None] = None,
        training_scheme: Union[str, None] = None,
        optim_name: Union[str, None] = None,
        optim_category: Union[str, None] = None,
        optim_target: Union[str, None] = None,
        release_version: Union[str, None] = None,
        page: int = 1,
        page_length: int = 20,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        force_token_refresh: bool = False,
    ) -> List:
        args = ModelArgs(
            domain=domain,
            sub_domain=sub_domain,
            architecture=architecture,
            sub_architecture=sub_architecture,
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            optim_name=optim_name,
            optim_category=optim_category,
            optim_target=optim_target,
            release_version=release_version,
        )
        response_json = search_get_request(
            args=args,
            page=page,
            page_length=page_length,
            force_token_refresh=force_token_refresh,
        )

        return [
            Model(
                **model,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
            )
            for model in response_json["models"]
        ]

    def __init__(
        self,
        display_name: str,
        display_description: str,
        files: List[Dict[str, Any]] = None,
        recipes: List[Dict[str, Any]] = None,
        results: List[Dict[str, Any]] = None,
        tags: List[Dict[str, Any]] = None,
        user: Dict[str, Any] = None,
        release_version: Dict[str, Any] = None,
        override_folder_name: Union[str, None] = None,
        override_parent_path: Union[str, None] = None,
        **kwargs,
    ):
        release_version = ReleaseVersion(**release_version) if release_version else None

        super(Model, self).__init__(
            **kwargs,
            folder_name=kwargs["model_id"]
            if not override_folder_name
            else kwargs["model_id"],
            override_parent_path=override_parent_path,
        )
        super(Downloadable, self).__init__(**kwargs, release_version=release_version)

        self._results = [Result(**res) for res in results] if results else []
        self._tags = [Tag(**res) for res in tags] if tags else []
        self._user = User(**user) if user else None
        self._release_version = release_version

        metadata = ModelMetadata(**kwargs, release_version=self._release_version)
        sorted_files = {
            file_type.value: [
                res for res in files if res["file_type"] == file_type.value
            ]
            for file_type in FileTypes
        }

        self._display_name = display_name
        self._display_description = display_description

        self._recipes = (
            [
                OptimizationRecipe(
                    model_metadata=metadata,
                    override_folder_name=override_folder_name,
                    override_parent_path=override_parent_path,
                    **res,
                )
                for res in recipes
            ]
            if recipes
            else []
        )

        self._data_inputs = (
            Data(
                name="inputs",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_INPUTS.value][0],
            )
            if sorted_files[FileTypes.DATA_INPUTS.value]
            else None
        )
        self._data_outputs = (
            Data(
                name="outputs",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_OUTPUTS.value][0],
            )
            if sorted_files[FileTypes.DATA_OUTPUTS.value]
            else None
        )
        self._data_labels = (
            Data(
                name="labels",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_LABELS.value][0],
            )
            if sorted_files[FileTypes.DATA_LABELS.value]
            else None
        )
        self._data_originals = (
            Data(
                name="originals",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_ORIGINALS.value][0],
            )
            if sorted_files[FileTypes.DATA_ORIGINALS.value]
            else None
        )

        self._framework_files = [
            File(
                model_metadata=metadata,
                child_folder_name=metadata.framework,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **res,
            )
            for res in sorted_files[FileTypes.FRAMEWORK.value]
        ]

        self._onnx_files = (
            [
                File(
                    model_metadata=metadata,
                    override_folder_name=override_folder_name,
                    override_parent_path=override_parent_path,
                    **res,
                )
                for res in sorted_files[FileTypes.ONNX.value]
            ]
            if sorted_files[FileTypes.ONNX.value]
            else None
        )
        if sorted_files[FileTypes.ONNX_GZ.value]:
            if not self._onnx_files:
                self._onnx_files = []
            self._onnx_files.extend(
                [
                    File(
                        model_metadata=metadata,
                        override_folder_name=override_folder_name,
                        override_parent_path=override_parent_path,
                        **res,
                    )
                    for res in sorted_files[FileTypes.ONNX_GZ.value]
                ]
            )

        self._card_file = (
            File(
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.CARD.value][0],
            )
            if sorted_files[FileTypes.CARD.value]
            else None
        )

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
    def recipes(self) -> List[OptimizationRecipe]:
        """
        :return: list of optimization recipes for the model
        """
        return self._recipes

    @property
    def data_inputs(self) -> Union[None, Data]:
        """
        :return: sample numpy data for inputs into the model
        """
        return self._data_inputs

    @property
    def data_outputs(self) -> Union[None, Data]:
        """
        :return: sample numpy data recorded as outputs from the model for a given input
        """
        return self._data_outputs

    @property
    def data_labels(self) -> Union[None, Data]:
        """
        :return: sample numpy data for labels for a given input
        """
        return self._data_labels

    @property
    def data_originals(self) -> Union[None, Data]:
        """
        :return: sample numpy data as originals before any pre processing is applied
            to the data to create the inputs for the model
        """
        return self._data_originals

    @property
    def framework_files(self) -> List[File]:
        """
        :return: list of Files that are of type framework
        """
        return self._framework_files

    @property
    def onnx_files(self) -> List[File]:
        """
        :return: list of Files that are of type onnx
        """
        return self._onnx_files

    @property
    def onnx_file(self) -> Union[File, None]:
        """
        :return: The latest converted onnx file for the model
        """
        onnx_file = None

        for file in self.onnx_files:
            if file.file_type == FileTypes.ONNX and file.display_name == "model.onnx":
                onnx_file = file

        return onnx_file

    @property
    def onnx_file_gz(self) -> Union[File, None]:
        """
        :return: The latest converted gziped onnx file for the model
        """
        onnx_file = None

        for file in self.onnx_files:
            if (
                file.file_type == FileTypes.ONNX_GZ
                and file.display_name == "model.onnx"
            ):
                onnx_file = file

        return onnx_file

    @property
    def card_file(self) -> Union[File, None]:
        """
        :return: The markdown card representing info about the model
        """
        return self._card_file

    @property
    def tags(self) -> List[Tag]:
        """
        :return: a list of model repo tags for this model
        """
        return self._tags

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

    def loader(
        self, batch_size: int, iter_steps: int = 0, batch_as_list: bool = True
    ) -> DataLoader:
        datasets = []

        if self.data_inputs:
            datasets.append(self.data_inputs.dataset())

        if self.data_outputs:
            datasets.append(self.data_outputs.dataset())

        if self.data_labels:
            datasets.append(self.data_labels.dataset())

        if self.data_originals:
            datasets.append(self.data_originals.dataset())

        if len(datasets) < 1:
            raise FileNotFoundError(
                "no datasets available for this model to create a loader from"
            )

        return DataLoader(
            *datasets,
            batch_size=batch_size,
            iter_steps=iter_steps,
            batch_as_list=batch_as_list,
        )

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
        _LOGGER.info(f"Downloading model {self.model_url_path}")

        if self.card_file:
            self.card_file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        if self.onnx_file:
            self.onnx_file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        if self.onnx_file_gz:
            self.onnx_file_gz.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        for file in self._framework_files:
            file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        for data in [
            self._data_inputs,
            self._data_outputs,
            self.data_labels,
            self.data_originals,
        ]:
            if data:
                data.download(
                    overwrite=overwrite,
                    refresh_token=refresh_token,
                    show_progress=show_progress,
                )

        for recipe in self._recipes:
            recipe.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )
