import pytest

from sparsezoo.models.classification import (
    resnet_18,
    resnet_34,
    resnet_50,
    resnet_50_2x,
    resnet_101,
    resnet_101_2x,
    resnet_152,
)
from tests.sparsezoo.utils import (
    ALL_MODELS_SKIP_MESSAGE,
    SPARSEZOO_TEST_ALL_RESNET,
    model_constructor,
)


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_RESNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_18(
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
        resnet_18,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_RESNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_34(
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
        resnet_34,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "conservative", None),
    ],
)
def test_resnet_50(
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
        resnet_50,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_RESNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
        (
            True,
            "pytorch",
            "sparseml",
            "imagenet",
            None,
            "pruned_quant",
            "conservative",
            None,
        ),
        (True, "tensorflow_v1", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenette", None, "base", "none", None),
        (
            True,
            "pytorch",
            "sparseml",
            "imagenette",
            None,
            "pruned",
            "conservative",
            None,
        ),
        (True, "pytorch", "sparseml", "imagenette", None, "pruned", "moderate", None),
        (
            True,
            "pytorch",
            "torchvision",
            "imagenette",
            None,
            "pruned",
            "conservative",
            None,
        ),
    ],
)
def test_resnet_50_extended(
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
        resnet_50,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_RESNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_50_2x(
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
        resnet_50_2x,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_RESNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_101(
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
        resnet_101,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_RESNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_101_2x(
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
        resnet_101_2x,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )


@pytest.mark.skipif(not SPARSEZOO_TEST_ALL_RESNET, reason=ALL_MODELS_SKIP_MESSAGE)
@pytest.mark.parametrize(
    (
        "download,framework,repo,dataset,training_scheme,"
        "optim_name,optim_category,optim_target"
    ),
    [
        (True, "pytorch", "sparseml", "imagenet", None, "base", "none", None),
        (True, "pytorch", "sparseml", "imagenet", None, "pruned", "moderate", None),
        (True, "pytorch", "torchvision", "imagenet", None, "base", "none", None),
    ],
)
def test_resnet_152(
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
        resnet_152,
        download,
        framework,
        repo,
        dataset,
        training_scheme,
        optim_name,
        optim_category,
        optim_target,
    )
