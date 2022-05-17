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

def get_zero_point(node: NodeProto):
    return node.input[2] if is_quantized_layer(node) else 0

def is_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if node weights have any sparsity, False otherwise
    """
    return get_param_sparsity(model, node) > 0

def is_four_block_sparse_layer(model: ModelProto, node: NodeProto) -> bool:
    """
    :return: True if node weights have any four block sparsity, False otherwise
    """
    return get_node_four_block_sparsity(model, node) > 0

def is_quantized_layer(node: NodeProto) -> bool:
    """
    :return: True if the node is not a float32 or float64, False otherwise
    """
    return node.op_type in ["ConvInteger", "MatMulInteger"]

def get_node_four_block_sparsity(param: numpy.ndarray) -> float:
    """
    :return: The sparsity of the parameter in terms of blocks of four
    """

    # TODO: More thought about matmul input channel dimension
    # TODO: vectorize

    zero_point = get_zero_point(node)
    param = get_layer_param(model, node)
    param_four_blocked = numpy.reshape(param, (-1, 4))

    num_zero_blocks = np.count_nonzero(param_four_blocked == [0, 0, 0, 0])

    return num_zero_blocks / param.size

def get_node_sparsity(model: ModelProto, node: NodeProto) -> float:
    """
    :return: The number proportion of zeros in the given parameter
    """
    zero_point = get_zero_point(node)
    param = get_layer_param(model, node)
    num_zeros = np.count_nonzero(param == zero_point)

    return num_zeros / param.size

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
        if initializer is None: raise Exception("Parameter not found")

        return numpy_helper.to_array(initializer)

    elif node.op_type in ["MatMul", "Gemm", "MatMulInteger"]:
        initializer_names = [init.name for init in model.graph.initializer]
        initializer_name = next(name for name in node.input
                                if name in initializer_names)

        initializer = get_initializer(model, initializer_name)
        if initializer is None: raise Exception("Parameter not found")

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
