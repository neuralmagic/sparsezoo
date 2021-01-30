import pytest

from sparsezoo.models.detection import ssd_resnet50_300
from tests.sparsezoo.utils import (
    ALL_MODELS_SKIP_MESSAGE,
    SPARSEZOO_TEST_ALL_SSD,
    model_constructor,
)


@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "coco", None, "base", "none", None),
        (True, "pytorch", "sparseml", "coco", None, "pruned", "moderate", None),
    ],
)
def test_ssd_resnet50_300(
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
        ssd_resnet50_300,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_SSD, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "voc", None, "base", "none", None),
        (True, "pytorch", "sparseml", "voc", None, "pruned", "moderate", None),
    ],
)
def test_ssd_resnet50_300_extended(
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
        ssd_resnet50_300,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )
