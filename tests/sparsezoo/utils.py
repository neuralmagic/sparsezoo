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

import os
from typing import List

from sparsezoo.objects import Model


def validate_downloaded_model(
    model: Model, check_model_args=None, check_other_args=None
):
    if check_model_args:
        for key, value in check_model_args.items():
            assert getattr(model, key) == value

    if check_other_args:
        if "override_parent_path" in check_other_args:
            assert check_other_args["override_parent_path"] in model.dir_path
        if "override_folder_name" in check_other_args:
            assert check_other_args["override_folder_name"] in model.dir_path

    assert os.path.exists(model.dir_path)
    assert os.path.exists(model.card_file.path)
    assert os.path.exists(model.onnx_file.path)

    assert len(model.framework_files) > 0
    for file in model.framework_files:
        assert os.path.exists(file.path)

    assert len(model.recipes) > (0 if model.optim_name != "base" else -1)
    for recipe in model.recipes:
        assert os.path.exists(recipe.path)

    assert os.path.exists(model.data_inputs.path)
    assert os.path.exists(model.data_outputs.path)

    num_batches = 0
    for batch in model.data_loader(batch_size=16, iter_steps=5):
        assert "inputs" in batch
        assert "outputs" in batch
        num_batches += 1
    assert num_batches == 5

    for data_name, data in model.data.items():
        batch = data.sample_batch()
        assert batch
        assert isinstance(batch, List)
