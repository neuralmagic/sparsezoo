import pytest

from sparsezoo.api import search_models


@pytest.mark.parametrize(
    "model_args,other_args",
    [
        ({"domain": "cv", "sub_domain": "classification"}, {}),
        ({"domain": "cv", "sub_domain": "classification"}, {"page_length": 1}),
        (
            {
                "domain": "cv",
                "sub_domain": "classification",
                "optimization_name": "base",
            },
            {},
        ),
    ],
)
def test_search_models(model_args, other_args):
    models = search_models(**model_args, **other_args)

    for model in models:
        for key, value in model_args.items():
            assert getattr(model, key) == value

    if "page_length" in other_args:
        assert len(models) <= other_args["page_length"]
