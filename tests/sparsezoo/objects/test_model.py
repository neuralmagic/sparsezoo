import os
import shutil
import pytest

from sparsezoo.objects import Model
from sparsezoo.utils import CACHE_DIR

from tests.sparsezoo.utils import validate_model_downloaded


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        ({"domain": "cv", "sub_domain": "classification"}, {}),
        ({"domain": "cv", "sub_domain": "classification"}, {"page_length": 1}),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
            },
            {},
        ),
        ({"domain": "cv", "sub_domain": "classification", "optim_name": "base",}, {},),
    ],
)
def test_model_search_downloadable(model_args, other_args):
    models = Model.search_downloadable(**model_args, **other_args)

    for model in models:
        for key, value in model_args.items():
            assert getattr(model, key) == value

    if "page_length" in other_args:
        assert len(models) <= other_args["page_length"]


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "torchvision",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {
                "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
                "override_folder_name": "test_folder",
            },
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "framework": "pytorch",
                "repo": "torchvision",
                "dataset": "imagenet",
                "training_scheme": None,
                "optim_name": "base",
                "optim_category": "none",
                "optim_target": None,
            },
            {},
        ),
    ],
)
def test_model_get_downloadable(model_args, other_args):
    model = Model.get_downloadable(**model_args, **other_args)
    model.download(overwrite=True)
    validate_model_downloaded(model, model_args, other_args)
    shutil.rmtree(model.dir_path)
