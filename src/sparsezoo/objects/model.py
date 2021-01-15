"""
Code related to a model from the sparsezoo
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Union

from sparsezoo.objects.data import Data
from sparsezoo.objects.downloadable import Downloadable
from sparsezoo.objects.file import File, FileTypes
from sparsezoo.objects.metadata import ModelMetadata, OptimizationId
from sparsezoo.objects.optimization_recipe import OptimizationRecipe
from sparsezoo.objects.release_version import ReleaseVersion
from sparsezoo.objects.result import Result
from sparsezoo.objects.tag import Tag
from sparsezoo.objects.user import User
from sparsezoo.requests import ModelArgs, download_get_request, search_get_request
from sparsezoo.utils import DataLoader


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
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
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

        :param domain: The domain of the model the object belongs to;
            e.g. cv, nlp
        :param sub_domain: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        :param architecture: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        :param sub_architecture: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        :param framework: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param repo: The source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param dataset: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param training_scheme: The training scheme used on the model the object
            belongs to if any; e.g. augmented
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :param release_version: The sparsezoo release version for the model
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
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
            args=args,
            file_name=None,
            force_token_refresh=force_token_refresh,
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
        """
        Obtains a list of Models matching the search parameters

        :param domain: The domain of the model the object belongs to;
            e.g. cv, nlp
        :param sub_domain: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        :param architecture: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        :param sub_architecture: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        :param framework: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param repo: The source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param dataset: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param training_scheme: The training scheme used on the model the object
            belongs to if any; e.g. augmented
        :param optim_name: The name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param optim_category: The degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param optim_target: The deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :param release_version: The sparsezoo release version for the model
        :param page: the page of values to get
        :param page_length: the page length of values to get
        :param override_folder_name: Override for the name of the folder to save
            this file under
        :param override_parent_path: Path to override the default save path
            for where to save the parent folder for this file under
        :param force_token_refresh: True to refresh the auth token, False otherwise
        :return: The requested Model instance
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
            else override_folder_name,
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

        self._data = OrderedDict()
        if sorted_files[FileTypes.DATA_INPUTS.value]:
            self._data["inputs"] = Data(
                name="inputs",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_INPUTS.value][0],
            )
        if sorted_files[FileTypes.DATA_OUTPUTS.value]:
            self._data["outputs"] = Data(
                name="outputs",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_OUTPUTS.value][0],
            )
        if sorted_files[FileTypes.DATA_LABELS.value]:
            self._data["labels"] = Data(
                name="labels",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_LABELS.value][0],
            )
        if sorted_files[FileTypes.DATA_ORIGINALS.value]:
            self._data["originals"] = Data(
                name="originals",
                model_metadata=metadata,
                override_folder_name=override_folder_name,
                override_parent_path=override_parent_path,
                **sorted_files[FileTypes.DATA_ORIGINALS.value][0],
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
        return self._data["inputs"] if "inputs" in self._data else None

    @property
    def data_outputs(self) -> Union[None, Data]:
        """
        :return: sample numpy data recorded as outputs from the model for a given input
        """
        return self._data["outputs"] if "outputs" in self._data else None

    @property
    def data_labels(self) -> Union[None, Data]:
        """
        :return: sample numpy data for labels for a given input
        """
        return self._data["labels"] if "labels" in self._data else None

    @property
    def data_originals(self) -> Union[None, Data]:
        """
        :return: sample numpy data as originals before any pre processing is applied
            to the data to create the inputs for the model
        """
        return self._data["originals"] if "originals" in self._data else None

    @property
    def data(self) -> Dict[str, Data]:
        """
        :return: A dictionary containing all of the Data objects for this model
        """
        return self._data

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
            if file.file_type_onnx and file.display_name == "model.onnx":
                onnx_file = file

        return onnx_file

    @property
    def onnx_file_gz(self) -> Union[File, None]:
        """
        :return: The latest converted gziped onnx file for the model
        """
        onnx_file = None

        for file in self.onnx_files:
            if file.file_type_onnx_gz and file.display_name == "model.onnx":
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

    def data_loader(
        self, batch_size: int, iter_steps: int = 0, batch_as_list: bool = True
    ) -> DataLoader:
        """
        Create a  data loader containing all of the available data for this model

        :param batch_size: the size of batches to create for the iterator
        :param iter_steps: the number of steps (batches) to create.
            Set to -1 for infinite, 0 for running through the loaded data once,
            or a positive integer for the desired number of steps
        :param batch_as_list: True to create the items from each dataset
            as a list, False for an ordereddict
        :return: The created data loader
        """
        data = self.data
        datasets = []

        for _, dat in data.items():
            datasets.append(dat.dataset())

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
        Downloads a model repo file.

        :param overwrite: True to overwrite the file if it exists, False otherwise
        :param refresh_token: refresh the auth token
        :param show_progress: True to use tqdm for progress, False to not show
        """
        _LOGGER.info(f"Downloading model {self.model_url_path}")

        if self.card_file:
            _LOGGER.info(f"Downloading model card {self.model_url_path}")
            self.card_file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        if self.onnx_file:
            _LOGGER.info(f"Downloading model onnx {self.model_url_path}")
            self.onnx_file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        if self.onnx_file_gz:
            _LOGGER.info(f"Downloading model onnx gz {self.model_url_path}")
            self.onnx_file_gz.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        for file in self._framework_files:
            _LOGGER.info(
                f"Downloading model framework file "
                f"{file.display_name} {self.model_url_path}"
            )
            file.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        for data in self._data.values():
            _LOGGER.info(f"Downloading model data {data.name} {self.model_url_path}")
            data.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

        for recipe in self._recipes:
            _LOGGER.info(
                f"Downloading model recipe {recipe.display_name} {self.model_url_path}"
            )
            recipe.download(
                overwrite=overwrite,
                refresh_token=refresh_token,
                show_progress=show_progress,
            )

    def search_similar(
        self,
        match_domain: bool = True,
        match_sub_domain: bool = True,
        match_architecture: bool = True,
        match_sub_architecture: bool = True,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = False,
        match_optim_name: bool = False,
        match_optim_category: bool = False,
        match_optim_target: bool = False,
    ) -> List:
        """
        Search for similar models to the current one

        :param match_domain: True to match similar models to the current
            domain of the model the object belongs to; e.g. cv, nlp
        :param match_sub_domain: True to match similar models to the current
            sub domain of the model the object belongs to;
            e.g. classification, segmentation
        :param match_architecture: True to match similar models to the current
            architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        :param match_sub_architecture: True to match similar models to the current
            sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        :param match_framework: True to match similar models to the current
            framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param match_repo: True to match similar models to the current
            source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param match_dataset: True to match similar models to the current
            dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param match_training_scheme: True to match similar models to the current
            training scheme used on the model the object
            belongs to if any; e.g. augmented
        :param match_optim_name: True to match similar models to the current
            name describing the optimization of the model
            the object belongs to, e.g. base, pruned, pruned_quant
        :param match_optim_category: True to match similar models to the current
            degree of optimization of the model the object
            belongs to; e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        :param match_optim_target: True to match similar models to the current
            deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        :return: a list of models matching the current model, if any
        """
        return Model.search_downloadable(
            domain=self.domain if match_domain else None,
            sub_domain=self.sub_domain if match_sub_domain else None,
            architecture=self.architecture if match_architecture else None,
            sub_architecture=self.sub_architecture if match_sub_architecture else None,
            framework=self.framework if match_framework else None,
            repo=self.repo if match_repo else None,
            dataset=self.dataset if match_dataset else None,
            training_scheme=self.training_scheme if match_training_scheme else None,
            optim_name=self.optim_name if match_optim_name else None,
            optim_category=self.optim_category if match_optim_category else None,
            optim_target=self.optim_target if match_optim_target else None,
        )

    def search_optimized_versions(
        self,
        match_framework: bool = True,
        match_repo: bool = True,
        match_dataset: bool = True,
        match_training_scheme: bool = True,
    ) -> List[OptimizationId]:
        """
        Search for different available optimized versions based off of the current model

        :param match_framework: True to match similar models to the current
            framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        :param match_repo: True to match similar models to the current
            source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        :param match_dataset: True to match similar models to the current
            dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        :param match_training_scheme: True to match similar models to the current
            training scheme used on the model the object
            belongs to if any; e.g. augmented
        :return: the list of matching optimization ids, if any
        """
        matched = self.search_similar(
            match_domain=True,
            match_sub_domain=True,
            match_architecture=True,
            match_sub_architecture=True,
            match_framework=match_framework,
            match_repo=match_repo,
            match_dataset=match_dataset,
            match_training_scheme=match_training_scheme,
        )
        ids = []

        for match in matched:
            ids.append(
                OptimizationId(
                    match.optim_name, match.optim_category, match.optim_target
                )
            )

        return ids
