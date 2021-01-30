import pytest

from sparsezoo.models.classification import efficientnet_b0, efficientnet_b4
from tests.sparsezoo.utils import (
    ALL_MODELS_SKIP_MESSAGE,
    SPARSEZOO_TEST_ALL_EFFICIENTNET,
    model_constructor,
)


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_EFFICIENTNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "arch", "moderate", None),
    ],
)
def test_efficientnet_b0(
    download,
    framework,
    repo,
    dataset,
    training_scheme,
    optim_name,
    optim_category,
    optim_target,
):
    model_constructor(
        efficientnet_b0,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(
    not SPARSEZOO_TEST_ALL_EFFICIENTNET, reason="Not running full suite of tests"
)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "arch", "moderate", None),
    ],
)
def test_efficientnet_b4(
    download,
    framework,
    repo,
    dataset,
    training_scheme,
    optim_name,
    optim_category,
    optim_target,
):
    model_constructor(
        efficientnet_b4,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )
