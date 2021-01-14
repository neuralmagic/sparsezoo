"""
Inception models:
    https://arxiv.org/abs/1512.00567
"""

from typing import Union

from sparsezoo.objects import Model

__all__ = ["inception_v3"]


def inception_v3(
    framework: str = "pytorch",
    repo: str = "sparseml",
    dataset: str = "imagenet",
    training_scheme: Union[str, None] = None,
    optim_name: str = "base",
    optim_category: str = "none",
    optim_target: Union[str, None] = None,
    override_folder_name: Union[str, None] = None,
    override_parent_path: Union[str, None] = None,
    force_token_refresh: bool = False,
):
    """
    Convenience function for getting an inception_v3 model

    :param framework: The framework the model the object belongs to was trained on;
        e.g. pytorch, tensorflow
    :param repo: the source repo for the model the object belongs to;
        e.g. sparseml, torchvision
    :param dataset: The dataset the model the object belongs to was trained on;
        e.g. imagenet, cifar10
    :param training_scheme: The training scheme used on the model the object
        belongs to if any; e.g. augmented
    :param optim_name: The name describing the optimization of the model
        the object belongs to, e.g. base, sparse, sparse_quant
    :param optim_category: The degree of optimization of the model the object
        belongs to; e.g. none, conservative (~100% baseline metric),
        moderate (>=99% baseline metric), aggressive (<99% baseline metric)
    :param optim_target: The deployment target of optimization of the model
        the object belongs to; e.g. edge, deepsparse, deepsparse_throughput, gpu
    :param override_folder_name: Override for the name of the folder to save
        this file under
    :param override_parent_path: Path to override the default save path
        for where to save the parent folder for this file under
    :param force_token_refresh: True to refresh the auth token, False otherwise
    :return: The created model
    """
    return Model.get_downloadable(
        domain="cv",
        sub_domain="classification",
        architecture="inception_v3",
        sub_architecture=None,
        framework=framework,
        repo=repo,
        dataset=dataset,
        training_scheme=training_scheme,
        optim_name=optim_name,
        optim_category=optim_category,
        optim_target=optim_target,
        override_folder_name=override_folder_name,
        override_parent_path=override_parent_path,
        force_token_refresh=force_token_refresh,
    )
