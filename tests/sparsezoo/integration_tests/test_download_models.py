import os
import shutil

import pytest
from sparsezoo.api import download_model, download_model_file
from sparsezoo.utils import CACHE_DIR


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "dataset": "imagenet",
                "repo_source": "torchvision",
                "framework": "pytorch",
                "optimization_name": "base",
                "file_name": "model.onnx",
            },
            {"save_dir": os.path.join(CACHE_DIR, "test_download")},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "dataset": "imagenet",
                "repo_source": "torchvision",
                "framework": "pytorch",
                "optimization_name": "base",
                "file_name": "model.onnx",
            },
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "dataset": "imagenet",
                "repo_source": "torchvision",
                "framework": "pytorch",
                "optimization_name": "base",
                "file_name": "model.onnx",
            },
            {"save_path": os.path.join(CACHE_DIR, "test_download", "name.onnx")},
        ),
    ],
)
def test_download_model_file(model_args, other_args):
    model_file, save_path = download_model_file(**model_args, **other_args)

    if "save_dir" in other_args:
        path = os.path.join(other_args["save_dir"], model_file.display_name)
    elif "save_path" in other_args:
        path = other_args["save_path"]
    else:
        path = os.path.join(CACHE_DIR, model_file.model_id, model_file.display_name)
    assert save_path == path
    assert os.path.exists(path)
    os.remove(path)
    assert not os.path.exists(path)


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "dataset": "imagenet",
                "repo_source": "torchvision",
                "framework": "pytorch",
                "optimization_name": "base",
            },
            {"save_dir": os.path.join(CACHE_DIR, "test_model_download")},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "dataset": "imagenet",
                "repo_source": "torchvision",
                "framework": "pytorch",
                "optimization_name": "base",
            },
            {},
        ),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "mobilenet_v1",
                "sub_architecture": "1.0",
                "dataset": "imagenet",
                "repo_source": "torchvision",
                "framework": "pytorch",
                "optimization_name": "base",
            },
            {"save_path": os.path.join(CACHE_DIR, "test_model_download", "mobilenet")},
        ),
    ],
)
def test_download_model(model_args, other_args):
    model, save_path = download_model(**model_args, **other_args)

    for key, value in model_args.items():
        assert getattr(model, key) == value

    if "save_dir" in other_args:
        root_path = os.path.join(other_args["save_dir"], model.model_id)
    elif "save_path" in other_args:
        root_path = other_args["save_path"]
    else:
        root_path = os.path.join(CACHE_DIR, model.model_id)
    assert os.path.abspath(save_path) == os.path.abspath(root_path)
    assert os.path.exists(root_path)
    for model_file in model.files:
        assert os.path.exists(os.path.join(root_path, model_file.display_name))

    shutil.rmtree(root_path)
