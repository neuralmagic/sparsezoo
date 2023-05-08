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

import json
import os
import tempfile
from collections import OrderedDict

import numpy as np
import onnx
import pytest
import yaml
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

import matplotlib.pyplot as plt
from sparsezoo.objects import File
from sparsezoo.utils import save_numpy, save_onnx, validate_onnx


MODEL_CARD = """
---
domain: "nlp"
sub_domain: "question_answering"
architecture: "bert"
sub_architecture: "base"
framework: "pytorch"
---
"""


def _create_yaml_file(file_path):
    test_dict = {"test_key": "test_value"}
    with open(file_path, "w") as outfile:
        yaml.dump(test_dict, outfile, default_flow_style=False)


def _create_numpy_file(file_path):
    numpy_input = np.array([1, 2, 3])
    numpy_dict = OrderedDict({"input_0": numpy_input, "input_1": numpy_input})
    file_path, _ = os.path.splitext(
        file_path
    )  # remove extension from the path (required by `save_numpy` function)
    save_numpy(
        array=numpy_dict,
        export_dir=os.path.dirname(file_path),
        name=os.path.basename(file_path),
    )


def _create_onnx_file(file_path):
    node = make_node("MatMul", ["input_0", "input_1"], ["output"], name="test_node")
    graph = make_graph(
        [node],
        "test_graph",
        [
            make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("input_1", onnx.TensorProto.FLOAT, (2, 1)),
        ],
        [make_tensor_value_info("output", onnx.TensorProto.FLOAT, (1, 1))],
    )
    model = make_model(graph)
    validate_onnx(model)
    save_onnx(model, file_path)


def _create_md_file(file_path):
    with open(file_path, "w") as outfile:
        outfile.write(MODEL_CARD)


def _create_json_file(file_path):
    json_string = "test_string"
    with open(file_path, "w") as outfile:
        json.dump(json_string, outfile)


def _create_image_file(file_path):
    image_array = np.zeros((10, 10, 3), dtype=np.uint8)
    plt.imsave(file_path, image_array)


def _create_csv_file(file_path):
    test_string = "test_string"
    with open(file_path, "w") as outfile:
        outfile.write(test_string)


def _create_sample_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension == ".npz":
        _create_numpy_file(file_path)
    elif extension == ".onnx":
        _create_onnx_file(file_path)
    elif extension == ".md":
        _create_md_file(file_path)
    elif extension == ".json":
        _create_json_file(file_path)
    elif extension in [".jpg", ".png", ".jpeg"]:
        _create_image_file(file_path)
    elif extension == ".csv":
        _create_csv_file(file_path)
    elif extension == ".yaml":
        _create_yaml_file(file_path)


@pytest.mark.parametrize(
    "extension, is_loadable",
    [
        (".npz", True),
        (".onnx", True),
        (".md", True),
        (".json", True),
        (".csv", True),
        (".jpg", True),
        (".png", True),
        (".jpeg", True),
        ("", True),
        (".bin", True),
    ],
    scope="function",
)
class TestFile:
    @pytest.fixture()
    def setup(self, extension, is_loadable):
        # setup
        _, path = tempfile.mkstemp(suffix=extension)
        _create_sample_file(path)

        yield path, is_loadable

        # teardown
        os.remove(path)

    def test_validate(self, setup):
        path, is_loadable = setup
        name = os.path.basename(path)
        file = File(name=name, path=path)
        assert is_loadable == file.validate()
