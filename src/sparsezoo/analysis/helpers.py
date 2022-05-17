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
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

def is_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if node weights have any sparsity, False otherwise
    """
    param = get_layer_param(model, node)
    sparsity = get_param_sparsity(param)

    return sparsity > 0

def is_four_block_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if node weights have any four block sparsity, False otherwise
    """
    param = get_layer_param(model, node)
    four_block_sparsity = get_param_four_block_sparsity(param)

    return four_block_sparsity > 0

def is_quantized_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if the node is not a float32 or float64, False otherwise
    """
    # TODO: Reimplement
    param = get_layer_param(model, node)
    return param.dtype in [numpy.float32, numpy.float64]

def get_param_four_block_sparsity(param: numpy.ndarray) -> float:
    """
    :return: The sparsity of the parameter in terms of blocks of four
    """

    # matmul input channel dimension
    # TODO: vectorize
    # TODO: quantized case

    param_flattened = param.flatten()
    if len(param_flattened) % 4 != 0:
        raise Exception("Parameter shape is not divisible by four")

    num_nonzero_blocks = 0
    for block_i in range(0, len(param_flattened), 4):
        block = param_flattened[block_i: block_i + 4]

        if any(block):
            num_nonzero_blocks += 1

    return 1 - num_nonzero_blocks / (len(param_flattened) / 4)

def get_param_sparsity(param: numpy.ndarray) -> float:
    """
    :return: The number proportion of zeros in the given parameter
    """

    # TODO: vectorize
    # TODO: quantized case

    param_flattened = param.flatten()
    return 1 - numpy.count_nonzero(param_flattened) / len(param_flattened)

def is_parameterized_prunable_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if this node performs a operation that is parameterized and
        prunable, False otherwise
    """
    return node.op_type in ["Conv", "MatMul", "Gemm", "QLinearConv",
                            "QLinearMatMul", "ConvInteger", "MatMulInteger",
                            "Gather"] and not get_layer_param(model, node) is None

# This is where I will handle all my edge cases
def get_layer_param(model: ModelProto, node: NodeProto) -> numpy.ndarray:
    """
    Finds the parameter value of the node. May raise error if param is not found
    :return: A numpy array of the param value
    """
    def get_initializer(model: ModelProto, initializer_name: str) -> numpy.ndarray:
        """
        Finds the initializer whose name is initalizer_name in the model graph
        :return: The initalizer if found, None otherwise
        """
        for initializer in model.graph.initializer:
            if initializer.name == initializer_name:
                return initializer

        return None

    if node.op_type == "Conv":
        initializer_name = node.input[1]

        initializer = get_initializer(model, initializer_name)
        if initializer is None: raise KeyError("Parameter not found")

        return numpy_helper.to_array(initializer)

    elif node.op_type in ["MatMul", "Gemm"]:
        initializer_names = [init.name for init in model.graph.initializer]
        initializer_name = next(name for name in node.input
                                if name in initializer_names)

        initializer = get_initializer(model, initializer_name)
        if initializer is None: raise KeyError("Parameter not found")

        return numpy_helper.to_array(initializer)

    else:
        return None

def get_layer_and_op_counts(model: ModelProto):
    """
    Creates two dictionaries, each mapping op_type to the number of nodes of
        that op_type. The first dictionary contains op_types which are layers,
        the second contains op_types which are operations.
    :return: a layer dictionary and an operation dictionary which hold node counts
    """
    model_op_types = [node.op_type for node in model.graph.node]

    layer_counts = {}
    op_counts = {}
    for op_type in model_op_types:
        op_type_nodes = [node for node in model.graph.node if node.op_type == op_type]
        op_count = len(op_type_nodes)
        assert len(op_type_nodes) > 0

        if is_parameterized_prunable_layer(model, op_type_nodes[0]):
            layer_counts[op_type] = op_count

        else:
            op_counts[op_type] = op_count

    return layer_counts, op_counts
