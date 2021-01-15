import os
import shutil

import pytest
from sparsezoo.models.classification import mobilenet_v1
from sparsezoo.utils import CACHE_DIR
from tests.sparsezoo.utils import validate_downloaded_model


@pytest.mark.parametrize(
    "download,framework,repo,dataset,training_scheme,"
    "optim_name,optim_category,optim_target",
    [
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_constructor(
    download,
    framework,
    repo,
    dataset,
    training_scheme,
    optim_name,
    optim_category,
    optim_target,
):
    other_args = {
        "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
    }

    if framework is None:
        model = mobilenet_v1(**other_args)
    else:
        model = mobilenet_v1(
            framework,
            repo,
            dataset,
            training_scheme,
            optim_name,
            optim_category,
            optim_target,
            **other_args,
        )
    assert model

    if download:
        model.download(overwrite=True)
        validate_downloaded_model(model, check_other_args=other_args)
        shutil.rmtree(model.dir_path)
