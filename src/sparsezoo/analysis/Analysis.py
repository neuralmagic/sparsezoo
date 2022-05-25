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

from functools import partial
from typing import Dict, List, Optional

import numpy as np
import onnx
from onnx import ModelProto

from pydantic import BaseModel, Field
from sparsezoo.analysis.helpers import (
    get_layer_and_op_counts,
    get_layer_param,
    get_node_four_block_sparsity,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_sparsity,
    get_zero_point,
    is_parameterized_prunable_layer,
    is_quantized_layer,
    is_sparse_layer,
)


__all__ = [
    "NodeAnalysis",
    "ModelAnalysis",
]


def map_and_accumulate(array, map_fn, acc_fn):
    """
    Helper function to apply a map function to each element in the array,
    then apply the accumulate function to the whole array

    :param: array to apply map function to
    :param: map function to apply to the array
    :param: accumulator function to apply after mapping

    :return: The result of applying map function and then accumulator function
    """
    return acc_fn(list(map(map_fn, array)))


class NodeAnalysis(BaseModel):
    """
    Pydantic model for the analysis of a node within a model
    """

    name: str = Field(description="This node's name")
    sparsity: float = Field(description="Proportion of zeros in this node's parameter")
    four_block_sparsity: float = Field(
        description="Proportion of zero blocks in this node's parameter"
    )
    param_size: int = Field(description="Size of this node's parameter")
    num_sparse_values: int = Field(
        description="Number of zeros in this node's parameter"
    )
    num_four_blocks: int = Field(
        description="Number of four blocks in this node's parameter"
    )
    num_sparse_four_blocks: int = Field(
        description="Number of four blocks with all zeros in this node's parameter"
    )
    quantized_layer: bool = Field(description="Does this node have quantized weights")
    zero_point: int = Field(
        description="Node zero point for quantization, default zero"
    )
    dtype: Optional[str] = Field(
        description="This node's parameter's data type", default=None
    )


class ModelAnalysis(BaseModel):
    """
    Pydantic model for analysis of a model
    """

    layer_counts: Dict[str, int] = Field(description="Overview of layers", default={})
    op_counts: Dict[str, int] = Field(
        description="Overview of operations (nodes that are not layers)", default={}
    )
    num_nodes: int = Field(description="Total number of nodes")
    num_layers: int = Field(description="Total number of layers")
    num_operations: int = Field(
        description="Total number of operations (nodes that are not layers)"
    )
    average_sparsity: float = Field(
        description="Average sparsity across all trainable parameters "
        "(excluding biases)"
    )
    average_four_block_sparsity: float = Field(
        description="Average sparsity four block sparsity across all "
        "trainable parameters (excluding biases)"
    )
    num_sparse_layers: int = Field(description="Number of layers with any sparsity")
    num_quantized_layers: int = Field(description="Number of quantized layers")
    num_parameters: int = Field(
        description="Total number of all parameters (excluding biases)"
    )
    num_sparse_parameters: int = Field(
        description="Total number of all sparsified parameters (excluding biases)"
    )
    num_four_blocks: int = Field(
        description="Total number of all four blocks (excluding biases)"
    )
    num_sparse_four_blocks: int = Field(
        description="Total number of all sparsified four blocks (excluding biases)"
    )

    nodes: List[NodeAnalysis] = Field(description="List of node analyses", default=[])

    @classmethod
    def from_onnx_model(cls, onnx_file_path: str):
        """
        Class constructor for Model Analysis

        :param cls: class being constructed
        :param onnx_file_path: path to onnx file being analyzed
        :return: instance of ModelAnalysis class
        """
        model_onnx = onnx.load(onnx_file_path)

        param_prunable_nodes = list(
            filter(
                partial(is_parameterized_prunable_layer, model_onnx),
                model_onnx.graph.node,
            )
        )
        sparsity_sizes = np.array(
            [
                get_node_num_zeros_and_size(model_onnx, node)
                for node in param_prunable_nodes
            ]
        )
        four_block_sparsity_sizes = np.array(
            [
                get_node_num_four_block_zeros_and_size(model_onnx, node)
                for node in param_prunable_nodes
            ]
        )

        num_sparse_parameters, num_parameters = np.sum(sparsity_sizes, axis=0)
        num_sparse_four_blocks, num_four_blocks = np.sum(
            four_block_sparsity_sizes, axis=0
        )

        layer_counts, op_counts = get_layer_and_op_counts(model_onnx)
        num_nodes = len(model_onnx.graph.node)
        num_layers = sum(layer_counts.values())
        num_operations = sum(op_counts.values())
        num_sparse_layers = map_and_accumulate(
            model_onnx.graph.node, partial(is_sparse_layer, model_onnx), sum
        )
        num_quantized_layers = map_and_accumulate(
            model_onnx.graph.node, partial(is_quantized_layer, model_onnx), sum
        )
        average_sparsity = num_sparse_parameters / num_parameters
        average_four_block_sparsity = num_sparse_four_blocks / num_four_blocks
        nodes = cls.analyze_nodes(model_onnx)

        return cls(
            layer_counts=layer_counts,
            op_counts=op_counts,
            num_nodes=num_nodes,
            num_layers=num_layers,
            num_operations=num_operations,
            num_sparse_layers=num_sparse_layers,
            num_quantized_layers=num_quantized_layers,
            num_parameters=num_parameters,
            num_sparse_parameters=num_sparse_parameters,
            num_four_blocks=num_four_blocks,
            num_sparse_four_blocks=num_sparse_four_blocks,
            average_sparsity=average_sparsity,
            average_four_block_sparsity=average_four_block_sparsity,
            nodes=nodes,
        )

    def analyze_nodes(model_onnx: ModelProto) -> List[NodeAnalysis]:
        """
        :param: model that contains the nodes to be analyzed
        :return: List of node analyses from model graph
        """
        nodes = []
        for node in model_onnx.graph.node:
            node_zero_point = get_zero_point(model_onnx, node)
            num_sparse_values, node_num_values = get_node_num_zeros_and_size(
                model_onnx, node
            )
            (
                num_sparse_four_blocks,
                node_num_blocks,
            ) = get_node_num_four_block_zeros_and_size(model_onnx, node)
            node_param = get_layer_param(model_onnx, node)
            node_analysis = NodeAnalysis(
                name=node.name,
                sparsity=get_node_sparsity(model_onnx, node),
                four_block_sparsity=get_node_four_block_sparsity(model_onnx, node),
                param_size=node_num_values,
                num_sparse_values=num_sparse_values,
                num_four_blocks=node_num_blocks,
                num_sparse_four_blocks=num_sparse_four_blocks,
                quantized_layer=is_quantized_layer(model_onnx, node),
                zero_point=node_zero_point,
                dtype=str(node_param.dtype) if node_param is not None else None,
            )
            nodes.append(node_analysis)

        return nodes
