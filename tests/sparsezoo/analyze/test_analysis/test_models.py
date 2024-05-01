from sparsezoo.analyze_v1.utils.models import DenseSparseOps, ZeroNonZeroParams
import pytest

@pytest.mark.parametrize("model", [DenseSparseOps, ZeroNonZeroParams])
@pytest.mark.parametrize("computed_fields", [
    ["sparsity"]
    ]
                         )
def test_model_dump_has_computed_fields(model, computed_fields):
    model = model()
    model_dict = model.model_dump()
    for computed_field in computed_fields:
        assert computed_field in model_dict
        assert model_dict[computed_field] == getattr(model, computed_field)    