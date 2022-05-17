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

import numpy
import onnx
from onnx import numpy_helper

# Just some conceptual notes for myself
# A layer is any node that has initializers
# An operation is any node that does not have initializers
# A parameter is a node's first initalizer

def has_parameters(model: onnx.onnx_ml_pb2.ModelProto,
                   node: onnx.onnx_ml_pb2.NodeProto) -> bool:
    """
    :return: True if the node contains an initalizer input, False otherwise
    """
    for input_name in node.input:
        if not get_initializer_value(model, input_name) is None:
            return True

    return False

def is_parameterized_prunable_layer(model: onnx.onnx_ml_pb2.ModelProto,
                                    node: onnx.onnx_ml_pb2.NodeProto) -> bool:
    """
    :return: True if this node performs a operation that is parameterized and
        prunable, False otherwise
    """
    return node.op_type in ["Conv", "MatMul", "Gemm", "QLinearConv",
                            "QLinearMatMul", "ConvInteger", "MatMulInteger",
                            "Gather"] and has_parameters(model, node)

# Not sure whether to raise error if value not found
def get_initializer_value(model: onnx.onnx_ml_pb2.ModelProto,
                          initializer_name: str) -> numpy.ndarray:
    """
    Finds the initializer whose name is initalizer_name in the model graph
    :return: The initalizer if found, None otherwise
    """
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            return initializer

    return None

def get_layer_param(model: onnx.onnx_ml_pb2.ModelProto,
                    node: onnx.onnx_ml_pb2.NodeProto) -> numpy.ndarray:
    """
    Finds the parameter value of the node. May raise error if param is not found
    :return: A numpy array of the param value
    """
    if node.op_type == "Conv":
        initializer_name = node.input[1]

        initializer_value = get_initializer_value(model, initializer_name)
        if initializer_value is None: raise KeyError("Parameter not found")

        return numpy_helper.to_array(initializer_value)

    elif node.op_type in ["MatMul", "Gemm"]:
        initializer_names = [init.name for init in model.graph.initializer]
        initializer_name = next(name for name in node.input
                                if name in initializer_names)

        initializer_value = get_initializer_value(model, initializer_name)
        if initializer_value is None: raise KeyError("Parameter not found")

        return numpy_helper.to_array(initializer_value)

    else:
        raise Exception(f"Unsupported op type {node.op_type}")

def get_layer_and_op_counts(model: onnx.onnx_ml_pb2.ModelProto):
    """
    :return: Two dictionaries, each mapping op_type to the number of nodes with
        that op_type. The first dictionary contains op_types which are layers,
        the second contains op_types which are operations.
    """
    model_op_types = [node.op_type for node in model.graph.node]

    layer_dict = {}
    op_dict = {}
    for op_type in model_op_types:
        op_type_nodes = [node for node in model.graph.node if node.op_type == op_type]
        op_count = len(op_type_nodes)
        assert len(op_type_nodes) > 0

        if has_parameters(model, op_type_nodes[0]):
            layer_dict[op_type] = op_count

        else:
            op_dict[op_type] = op_count

    return layer_dict, op_dict
