import pytest

from sparsezoo.models.classification import inception_v3
from tests.sparsezoo.utils import (
    ALL_MODELS_SKIP_MESSAGE,
    SPARSEZOO_TEST_ALL_INCEPTION,
    model_constructor,
)


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_INCEPTION, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
    ],
)
def test_inception_v3(
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
        inception_v3,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )
