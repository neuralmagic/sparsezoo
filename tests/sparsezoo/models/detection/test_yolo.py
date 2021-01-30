import pytest

from sparsezoo.models.detection import yolo_v3
from tests.sparsezoo.utils import (
    ALL_MODELS_SKIP_MESSAGE,
    SPARSEZOO_TEST_ALL_YOLO,
    model_constructor,
)


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_YOLO, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "ultralytics", "coco", None, "base", "none", None),
        (True, "pytorch", "ultralytics", "coco", None, "pruned", "moderate", None),
    ],
)
def test_yolo_v3(
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
        yolo_v3,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )
