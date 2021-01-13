"""
Code related to base functionality for making requests
"""

from typing import Union, List

__all__ = ["BASE_API_URL", "ModelArgs"]

BASE_API_URL = "https://api.neuralmagic.com/models"


class ModelArgs:
    """
    [TODO]
    """

    def __init__(
        self,
        domain: Union[str, None],
        sub_domain: Union[str, None],
        architecture: Union[str, None],
        sub_architecture: Union[str, None],
        framework: Union[str, None],
        repo: Union[str, None],
        dataset: Union[str, None],
        training_scheme: Union[str, None],
        optim_name: Union[str, None],
        optim_category: Union[str, None],
        optim_target: Union[str, None],
        release_version: Union[str, None],
        **kwargs,
    ):
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
    def domain(self) -> Union[str, None]:
        """
        :return: The domain of the model the object belongs to;
            e.g. cv, nlp
        """
        return self._domain

    @property
    def sub_domain(self) -> Union[str, None]:
        """
        :return: The sub domain of the model the object belongs to;
            e.g. classification, segmentation
        """
        return self._sub_domain

    @property
    def architecture(self) -> Union[str, List[str], None]:
        """
        :return: The architecture of the model the object belongs to;
            e.g. resnet_v1, mobilenet_v1
        """
        return self._architecture

    @property
    def sub_architecture(self) -> Union[str, List[str], None]:
        """
        :return: The sub architecture (scaling factor) of the model
            the object belongs to; e.g. 50, 101, 152
        """
        return self._sub_architecture

    @property
    def framework(self) -> Union[str, List[str], None]:
        """
        :return: The framework the model the object belongs to was trained on;
            e.g. pytorch, tensorflow
        """
        return self._framework

    @property
    def repo(self) -> Union[str, List[str], None]:
        """
        :return: the source repo for the model the object belongs to;
            e.g. sparseml, torchvision
        """
        return self._repo

    @property
    def dataset(self) -> Union[str, List[str], None]:
        """
        :return: The dataset the model the object belongs to was trained on;
            e.g. imagenet, cifar10
        """
        return self._dataset

    @property
    def training_scheme(self) -> Union[str, List[str], None]:
        """
        :return: The training scheme used on the model the object belongs to if any;
            e.g. augmented
        """
        return self._training_scheme

    @property
    def optim_name(self) -> Union[str, List[str], None]:
        """
        :return: The name describing the optimization of the model
            the object belongs to, e.g. base, sparse, sparse_quant,
        """
        return self._optim_name

    @property
    def optim_category(self) -> Union[str, List[str], None]:
        """
        :return: The degree of optimization of the model the object belongs to;
            e.g. none, conservative (~100% baseline metric),
            moderate (>=99% baseline metric), aggressive (<99% baseline metric)
        """
        return self._optim_category

    @property
    def optim_target(self) -> Union[str, List[str], None]:
        """
        :return: The deployment target of optimization of the model
            the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
        """
        return self._optim_target

    @property
    def release_version(self) -> Union[str, None]:
        return self._release_version

    @property
    def architecture_id(self) -> str:
        if not self.architecture:
            return ""

        if not self.sub_architecture:
            return f"{self.architecture}"

        return f"{self.architecture}-{self.sub_architecture}"

    @property
    def training_id(self) -> str:
        if not self.dataset:
            return ""

        if not self.training_scheme:
            return f"{self.dataset}"

        return f"{self.dataset}-{self.training_scheme}"

    @property
    def optimization_id(self) -> str:
        if not self.optim_name:
            return ""

        if not self.optim_category:
            return f"{self.optim_name}"

        if not self.optim_target:
            return f"{self.optim_name}-{self.optim_category}"

        return f"{self.optim_name}-{self.optim_category}-{self.optim_target}"

    @property
    def model_url_root(self) -> str:
        if not self.domain:
            return ""

        if not self.sub_domain:
            return f"{self.domain}"

        return f"{self.domain}/{self.sub_domain}"

    @property
    def model_url_path(self) -> str:
        return "/".join(
            [
                self.model_url_root,
                self.architecture_id,
                f"{self.framework}" if self.framework else "",
                f"{self.repo}" if self.repo else "",
                self.training_id,
                self.optimization_id,
            ]
        )

    @property
    def model_url_args(self) -> List[str]:
        args = []

        for key in [
            "architecture",
            "sub_architecture",
            "framework",
            "repo",
            "dataset",
            "training_scheme",
            "optim_name",
            "optim_category",
            "optim_target",
        ]:
            value = getattr(self, key)

            if value and isinstance(value, List):
                args.extend([f"{key}={item}" for item in value])
            elif value:
                args.append(f"{key}={value}")

        return args
