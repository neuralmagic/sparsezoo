# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from sparsezoo.analyze_v1.utils.models import DenseSparseOps, ZeroNonZeroParams


@pytest.mark.parametrize("model", [DenseSparseOps, ZeroNonZeroParams])
@pytest.mark.parametrize("computed_fields", [["sparsity"]])
def test_model_dump_has_computed_fields(model, computed_fields):
    model = model()
    model_dict = model.model_dump()
    for computed_field in computed_fields:
        assert computed_field in model_dict
        assert model_dict[computed_field] == getattr(model, computed_field)
