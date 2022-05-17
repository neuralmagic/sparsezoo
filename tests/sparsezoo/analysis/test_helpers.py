import pytest

import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

from sparsezoo import Zoo
from sparsezoo.analysis import *

# Helper functions and dictionaries
model_stubs = {
    "yolact_none": "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none",
    "mobilenet_v1_pruned_moderate": "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/pruned-moderate",
    "bert_pruned_quantized": "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
}

# TODO: There might be a better way to code this that doesn't require writing the onnx to disk
model_onnxs = {}
for model_name, stub in model_stubs.items():
    model = Zoo.load_model_from_stub(stub)
    model.onnx_file.download()
    onnx_path = model.onnx_file.downloaded_path()

    model_onnx = onnx.load(onnx_path)
    model_onnxs[model_name] = model_onnx

def get_model_from_name(model_name):
    return model_onnxs[model_name]

def get_node_from_name(model, node_name):
    return [node for node in list(model.graph.node) if node.name == node_name][0]

### BEGIN TESTS ###

@pytest.mark.parametrize(
    "model_name,node_name,expected_shape",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_0", (32, 3, 3, 3)),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_16", None),
        ("mobilenet_v1_pruned_moderate", "Pad_82", None),
        ("mobilenet_v1_pruned_moderate", "AveragePool_83", None),
        ("mobilenet_v1_pruned_moderate", "Shape_84", None),
        ("mobilenet_v1_pruned_moderate", "Gather_86", None),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", None),
        ("mobilenet_v1_pruned_moderate", "Concat_88", None),
        ("mobilenet_v1_pruned_moderate", "Reshape_89", None),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", (1000, 1024)),
        ("mobilenet_v1_pruned_moderate", "Softmax_91", None),

        ("bert_pruned_quantized", "Gather_34", None),
        ("bert_pruned_quantized", "DequantizeLinear_27", None),
        ("bert_pruned_quantized", "MatMul_80_quant", (768, 768)),
        ("bert_pruned_quantized", "MatMul_157_quant", None),
    ]
)
def test_get_layer_param(model_name, node_name, expected_shape):
    model = get_model_from_name(model_name)
    node = get_node_from_name(model, node_name)

    param = get_layer_param(model, node)
    print(param)
    if expected_shape is None:
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

        ("bert_pruned_quantized", "Gather_34", False),
        ("bert_pruned_quantized", "DequantizeLinear_27", False),
        ("bert_pruned_quantized", "MatMul_80_quant", True),
        ("bert_pruned_quantized", "MatMul_157_quant", False),
    ]
)
def test_is_parameterized_prunable_layer(model_name, node_name, expected_bool):
    model = get_model_from_name(model_name)
    node = get_node_from_name(model, node_name)

    assert is_parameterized_prunable_layer(model, node) == expected_bool

@pytest.mark.parametrize(
    "model_name,expected_layer_counts,expected_op_counts",
    [
        ("mobilenet_v1_pruned_moderate", {'Conv': 27, 'Gemm': 1},
                                         {'BatchNormalization': 27, 'Relu': 27,
                                          'Constant': 2, 'Pad': 1, 'AveragePool': 1,
                                          'Shape': 1, 'Gather': 1, 'Unsqueeze': 1,
                                          'Concat': 1, 'Reshape': 1, 'Softmax': 1}),
        ("bert_pruned_quantized", {'MatMulInteger': 73},
                                  {'Unsqueeze': 99, 'Shape': 97, 'Gather': 100,
                                   'DequantizeLinear': 39, 'Cast': 74, 'Add': 174,
                                   'Sub': 26, 'Mul': 123, 'Slice': 1, 'ReduceMean': 50,
                                   'Pow': 25, 'Sqrt': 25, 'Div': 49, 'QuantizeLinear': 97,
                                   'Concat': 48, 'Reshape': 48, 'Transpose': 48,
                                   'Softmax': 12, 'Erf': 12, 'Split': 1, 'Squeeze': 2,
                                   'QLinearMatMul': 24}),
    ]
)
def test_get_layer_and_op_counts(model_name, expected_layer_counts, expected_op_counts):
    model = get_model_from_name(model_name)

    layer_counts, op_counts = get_layer_and_op_counts(model)
    print(get_layer_and_op_counts(model))
    assert layer_counts == expected_layer_counts
    assert op_counts == expected_op_counts

@pytest.mark.parametrize(
    "model_name,node_name,expected_bool",
    [
        ("mobilenet_v1_pruned_moderate", "Conv_72", False),
        ("mobilenet_v1_pruned_moderate", "BatchNormalization_79", False),
        ("mobilenet_v1_pruned_moderate", "Unsqueeze_87", False),
        ("mobilenet_v1_pruned_moderate", "Gemm_90", False),

        ("bert_pruned_quantized", "Gather_34", False),
        ("bert_pruned_quantized", "DequantizeLinear_27", False),
        ("bert_pruned_quantized", "MatMul_80_quant", True),
        ("bert_pruned_quantized", "MatMul_157_quant", False),
    ]
)
def test_is_quantized_layer(model_name, node_name, expected_bool):
    model = get_model_from_name(model_name)
    node = get_node_from_name(model, node_name)

    assert is_quantized_layer(node) == expected_bool

def test_get_zero_point():
    pass #get_zero_point

def test_get_node_sparsity():
    pass #get_node_sparsity

def test_is_sparse_layer():
    pass #is_sparse_layer
