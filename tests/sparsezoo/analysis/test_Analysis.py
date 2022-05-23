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

from sparsezoo import Zoo
from sparsezoo.analysis import ModelAnalysis, NodeAnalysis


model_stubs = {
    "yolact_none": "zoo:cv/segmentation/yolact-darknet53/"
    "pytorch/dbolya/coco/base-none",
    "mobilenet_v1_pruned_moderate": "zoo:cv/classification/mobilenet_v1-1.0/"
    "pytorch/sparseml/imagenet/pruned-moderate",
    "bert_pruned_quantized": "zoo:nlp/question_answering/bert-base/"
    "pytorch/huggingface/squad/"
    "12layer_pruned80_quant-none-vnni",
}

# TODO: There might be a better way to code this
#       that doesn't require writing the onnx to disk
model_onnx_paths = {}
for model_name, stub in model_stubs.items():
    model = Zoo.load_model_from_stub(stub)
    model.onnx_file.download()
    model_onnx_paths[model_name] = model.onnx_file.downloaded_path()

print(model_onnx_paths)


def get_onnx_path_from_model_name(model_name):
    return model_onnx_paths[model_name]


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 334),
        ("mobilenet_v1_pruned_moderate", 92),
        ("bert_pruned_quantized", 1247),
    ],
)
def test_num_nodes(model_name, expected_value):
    onnx_path = get_onnx_path_from_model_name(model_name)
    assert ModelAnalysis.from_onnx_model(onnx_path).num_nodes == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 85),
        ("mobilenet_v1_pruned_moderate", 28),
        ("bert_pruned_quantized", 73),
    ],
)
def test_num_layers(model_name, expected_value):
    onnx_path = get_onnx_path_from_model_name(model_name)
    assert ModelAnalysis.from_onnx_model(onnx_path).num_layers == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 249),
        ("mobilenet_v1_pruned_moderate", 64),
        ("bert_pruned_quantized", 1174),
    ],
)
def test_num_operations(model_name, expected_value):
    onnx_path = get_onnx_path_from_model_name(model_name)
    assert ModelAnalysis.from_onnx_model(onnx_path).num_operations == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0.0),
        ("mobilenet_v1_pruned_moderate", 0.6510255903416607),
        ("bert_pruned_quantized", 0.8114767024167978),
    ],
)
def test_average_sparsity(model_name, expected_value):
    onnx_path = get_onnx_path_from_model_name(model_name)
    assert ModelAnalysis.from_onnx_model(onnx_path).average_sparsity == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0.0),
        ("mobilenet_v1_pruned_moderate", 0.47080982863746257),
        ("bert_pruned_quantized", 0.7999834982006258),
    ],
)
def test_average_four_block_sparsity(model_name, expected_value):
    onnx_path = get_onnx_path_from_model_name(model_name)
    assert (
        ModelAnalysis.from_onnx_model(onnx_path).average_four_block_sparsity
        == expected_value
    )


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0),
        ("mobilenet_v1_pruned_moderate", 12),
        ("bert_pruned_quantized", 73),
    ],
)
def test_num_sparse_layers(model_name, expected_value):
    onnx_path = get_onnx_path_from_model_name(model_name)
    assert ModelAnalysis.from_onnx_model(onnx_path).num_sparse_layers == expected_value


@pytest.mark.parametrize(
    "model_name,expected_value",
    [
        ("yolact_none", 0),
        ("mobilenet_v1_pruned_moderate", 0),
        ("bert_pruned_quantized", 73),
    ],
)
def test_num_quantized_layers(model_name, expected_value):
    onnx_path = get_onnx_path_from_model_name(model_name)
    assert (
        ModelAnalysis.from_onnx_model(onnx_path).num_quantized_layers == expected_value
    )


@pytest.mark.parametrize(
    "model_name,expected_node_model",
    [
        (
            "yolact_none",
            NodeAnalysis(
                name="Conv_0",
                sparsity=0.0,
                four_block_sparsity=0.0,
                param_size=864,
                num_sparse_values=0,
                num_four_blocks=216,
                num_sparse_four_blocks=0,
                quantized_layer=False,
                zero_point=0,
            ),
        ),
        (
            "mobilenet_v1_pruned_moderate",
            NodeAnalysis(
                name="Conv_72",
                sparsity=0.8999996185302734,
                four_block_sparsity=0.6695098876953125,
                param_size=524288,
                num_sparse_values=471859,
                num_four_blocks=131072,
                num_sparse_four_blocks=87754,
                quantized_layer=False,
                zero_point=0,
            ),
        ),
        (
            "bert_pruned_quantized",
            NodeAnalysis(
                name="MatMul_80_quant",
                sparsity=0.8029022216796875,
                four_block_sparsity=0.7999945746527778,
                param_size=589824,
                num_sparse_values=473571,
                num_four_blocks=147456,
                num_sparse_four_blocks=117964,
                quantized_layer=True,
                zero_point=128,
            ),
        ),
    ],
)
def test_create_nodes(model_name, expected_node_model):
    onnx_path = get_onnx_path_from_model_name(model_name)
    nodes = ModelAnalysis.from_onnx_model(onnx_path).nodes
    found_nodes = [node for node in nodes if node.name == expected_node_model.name]
    assert len(found_nodes) == 1
    found_node = found_nodes[0]

    print(f"found node: {found_node}")
    print(f"expected_node_model: {expected_node_model}")

    assert found_node == expected_node_model
