from typing import Union

from sparsezoo.objects.base import BaseObject
from sparsezoo.objects.release_version import ReleaseVersion


__all__ = ["ModelMetadata"]


class ModelMetadata(BaseObject):
    """
    :param domain:
    :param sub_domain:
    :param architecture:
    :param sub_architecture:
    :param framework:
    :param repo:
    :param dataset:
    :param training_scheme:
    :param optim_name:
    :param optim_category:
    :param optim_target:
    :param user_id: the user id for the user who uploaded model
    :param release_version_id: the release version id for the release version of the
        model
    :param base_model: the model id of a model this model inherited from
    """

    def __init__(
        self,
        model_id: str,
        base_model: str,
        user_id: str,
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
        release_version: ReleaseVersion,
        **kwargs,
    ):
        super(ModelMetadata, self).__init__(**kwargs)
        self._model_id = model_id
        self._base_model = base_model
        self._user_id = user_id
        self._domain = domain
        self._sub_domain = sub_domain
        self._architecture = architecture
        self._sub_architecture = sub_architecture
        self._framework = framework
        self._repo = repo
        self._dataset = dataset
        self._training_scheme = training_scheme
        self._optim_name = optim_name
        self._optim_category = optim_category
        self._optim_target = optim_target
        self._release_version = release_version

    @property
    def model_id(self) -> str:
        """
        :return: The id of the model the object belongs to
        """
        return self._model_id

    @property
    def domain(self) -> str:
        """
        :return: The domain of the model the object belongs to;
            e.g. cv, nlp
        """
        return self._domain

    @property
    def sub_domain(self) -> str:
        """
        :return: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        """
        return self._sub_domain

    @property
    def architecture(self) -> str:
        """
        :return: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        """
        return self._architecture

    @property
    def sub_architecture(self) -> Union[str, None]:
        """
        :return: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        """
        return self._sub_architecture

    @property
    def framework(self) -> str:
        """
        :return: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        """
        return self._framework

    @property
    def repo(self) -> str:
        """
        :return: the source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        """
        return self._repo

    @property
    def dataset(self) -> str:
        """
        :return: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        """
        return self._dataset

    @property
    def training_scheme(self) -> Union[str, None]:
        """
        :return: The training scheme used on the model the object belongs to if any;
            e.g. augmented
        """
        return self._training_scheme

    @property
    def optim_name(self) -> str:
        """
        :return: The name describing the optimization of the model
            the object belongs to, e.g. base, sparse, sparse_quant,
        """
        return self._optim_name

    @property
    def optim_category(self) -> str:
        """
        :return: The degree of optimization of the model the object belongs to;
            e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        """
        return self._optim_category

    @property
    def optim_target(self) -> Union[str, None]:
        """
        :return: The deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        """
        return self._optim_target

    @property
    def release_version(self) -> ReleaseVersion:
        return self._release_version

    @property
    def architecture_id(self) -> str:
        return f"{self.architecture}-{self.sub_architecture}"

    @property
    def training_id(self) -> str:
        training_id = f"{self.dataset}"

        if self.training_scheme:
            training_id = f"{self.training_id}-{self.training_scheme}"

        return training_id

    @property
    def optimization_id(self) -> str:
        optimization_id = f"{self.optim_name}-{self.optim_category}"

        if self.optim_target:
            optimization_id = f"{optimization_id}-{self.optim_target}"

        return optimization_id

    @property
    def model_url_path(self) -> str:
        return "/".join(
            [
                self.domain,
                self.sub_domain,
                self.architecture_id,
                self.framework,
                self.repo,
                self.training_id,
                self.optimization_id,
            ]
        )
