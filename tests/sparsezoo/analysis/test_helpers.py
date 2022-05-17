import pytest

import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

from sparsezoo import Zoo
from sparsezoo.analysis import *

model_stubs = {
    "mobilenet_v1_pruned_moderate": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate"
}

# TODO: There might be a better way to code this that doesn't require writing the onnx to disk
model_onnxs = {}
for model_name, stub in model_stubs.items():
    model = Zoo.load_model_from_stub(stub)
    onnx_path = model.onnx_file.downloaded_path()

    model_onnx = onnx.load(onnx_path)
    model_onnxs[model_name] = model_onnx

### BEGIN TESTS ###

@pytest.mark.parametrize(
    "model_name,node_name,expected_shape,is_none",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", (32, 3, 3, 3), False),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", None, True),
        ("mobilenet_v1_pruned_moderate", "Pad_82", None, True),
        ("mobilenet_v1_pruned_moderate", "AveragePool_83", None, True),
        ("mobilenet_v1_pruned_moderate", "Shape_84", None, True),
        ("mobilenet_v1_pruned_moderate", "Gather_86", None, True),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", None, True),
        ("mobilenet_v1_pruned_moderate", "Concat_88", None, True),
        ("mobilenet_v1_pruned_moderate", "Reshape_89", None, True),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", (1000, 1024), False),
        ("mobilenet_v1_pruned_moderate", "Softmax_91", None, True),
    ]
)
def test_get_layer_param(model_name, node_name, expected_shape, is_none):
    model = model_onnxs[model_name]
    node = [node for node in list(model.graph.node) if node.name == node_name][0]

    param = get_layer_param(model, node)
    if is_none:
        assert param is None
    else:
        assert param.shape == expected_shape

@pytest.mark.parametrize(
    "model_name,node_name,expected_bool",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", True),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", False),
        ("mobilenet_v1_pruned_moderate", "Pad_82", False),
        ("mobilenet_v1_pruned_moderate", "AveragePool_83", False),
        ("mobilenet_v1_pruned_moderate", "Shape_84", False),
        ("mobilenet_v1_pruned_moderate", "Gather_86", False),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", False),
        ("mobilenet_v1_pruned_moderate", "Concat_88", False),
        ("mobilenet_v1_pruned_moderate", "Reshape_89", False),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", True),
        ("mobilenet_v1_pruned_moderate", "Softmax_91", False),
    ]
)
def test_is_parameterized_prunable_layer(model_name, node_name, expected_bool):
    model = model_onnxs[model_name]
    node = [node for node in list(model.graph.node) if node.name == node_name][0]

    assert is_parameterized_prunable_layer(model, node) == expected_bool

@pytest.mark.parametrize(
    "model_name,expected_layer_counts,expected_op_counts",
    [
        ("mobilenet_v1_pruned_moderate", {'Conv': 27, 'Gemm': 1},
                                         {'BatchNormalization': 27, 'Relu': 27,
                                          'Constant': 2, 'Pad': 1, 'AveragePool': 1,
                                          'Shape': 1, 'Gather': 1, 'Unsqueeze': 1,
                                          'Concat': 1, 'Reshape': 1, 'Softmax': 1}),
    ]
)
def test_get_layer_and_op_counts(model_name, expected_layer_counts, expected_op_counts):
    model = model_onnxs[model_name]

    layer_counts, op_counts = get_layer_and_op_counts(model)
    assert layer_counts == expected_layer_counts
    assert op_counts == expected_op_counts
