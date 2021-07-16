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
import shutil
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import onnxruntime
from sparsezoo.models import Zoo
from sparsezoo.objects import Model
from sparsezoo.utils import CACHE_DIR


__all__ = [
    "download_and_verify",
    "model_constructor",
    "validate_with_ort",
    "validate_downloaded_model",
]


def download_and_verify(model: str, other_args: Optional[Dict] = None):
    if other_args is None:
        other_args = {
            "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
        }
    model = Zoo.load_model_from_stub(model, **other_args)
    model.download(overwrite=True)
    validate_downloaded_model(model, check_other_args=other_args)
    shutil.rmtree(model.dir_path)


def model_constructor(
    constructor_function: Callable,
    download: bool,
    framework: str,
    repo: str,
    dataset: str,
    training_scheme: Optional[str],
    sparse_name: str,
    sparse_category: str,
    sparse_target: Optional[str],
):
    other_args = {
        "override_parent_path": os.path.join(CACHE_DIR, "test_download"),
    }

    if framework is None:
        model = constructor_function(**other_args)
    else:
        model = constructor_function(
            framework=framework,
            repo=repo,
            dataset=dataset,
            training_scheme=training_scheme,
            sparse_name=sparse_name,
            sparse_category=sparse_category,
            sparse_target=sparse_target,
            **other_args,
        )
    assert model

    if download:
        model.download(overwrite=True)
        validate_downloaded_model(model, check_other_args=other_args)
        shutil.rmtree(model.dir_path)


def validate_with_ort(path: str, input_data: List, output_data: List):
    sess_options = onnxruntime.SessionOptions()

    sess_options.log_severity_level = 3

    inf_session = onnxruntime.InferenceSession(
        path,
        sess_options,
    )

    sess_batch = {}

    for inp_index, inp in enumerate(inf_session.get_inputs()):
        sess_batch[inp.name] = input_data[inp_index]

    sess_outputs = [out.name for out in inf_session.get_outputs()]
    pred = inf_session.run(sess_outputs, sess_batch)
    pred_dict = OrderedDict((key, val) for key, val in zip(sess_outputs, pred))

    assert len(pred_dict.keys()) == len(output_data)
    for out_index, pred_key in enumerate(pred_dict.keys()):
        assert pred_dict[pred_key].shape == output_data[out_index].shape


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

    assert len(model.recipes) > (
        0 if (model.sparse_name != "base" and model.sparse_name != "arch") else -1
    )
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

    for batch in model.data_loader(batch_size=1, iter_steps=5):
        validate_with_ort(model.onnx_file.path, batch["inputs"], batch["outputs"])

    for data_name, data in model.data.items():
        batch = data.sample_batch()
        assert batch
        assert isinstance(batch, List)
