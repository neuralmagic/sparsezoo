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

from typing import Dict, List, Optional

import numpy
import onnx
from onnx import ModelProto

from pydantic import BaseModel, Field
from sparsezoo.analysis.helpers import (
    extract_node_shapes,
    get_layer_and_op_counts,
    get_node_four_block_sparsity,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_sparsity,
    get_node_weight,
    get_num_operations,
    get_zero_point,
    is_parameterized_prunable_layer,
    is_quantized_layer,
    is_sparse_layer,
)


__all__ = [
    "NodeAnalysis",
    "ModelAnalysis",
]


class NodeAnalysis(BaseModel):
    """
    Pydantic model for the analysis of a node within a model
    """

    name: str = Field(description="This node's name")
    num_ops: int = Field(
        description="The number of (floating point and/or integer) operations "
        "performed in this node"
    )
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
    operation_counts: Dict[str, int] = Field(
        description="Overview of operations (nodes that are not layers)", default={}
    )
    num_ops: int = Field(
        description="The total number of (floating point and/or integer) operations "
        "performed in one forward pass of the model"
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

    layers: List[NodeAnalysis] = Field(
        description="List of analyses for each layer in the model graph", default=[]
    )

    @classmethod
    def from_onnx_model(cls, onnx_file_path: str):
        """
        Class constructor for Model Analysis

        :param cls: class being constructed
        :param onnx_file_path: path to onnx file being analyzed
        :return: instance of ModelAnalysis class
        """
        model_onnx = onnx.load(onnx_file_path)

        param_prunable_nodes = [
            node
            for node in model_onnx.graph.node
            if is_parameterized_prunable_layer(model_onnx, node)
        ]
        sparsity_sizes = numpy.array(
            [
                get_node_num_zeros_and_size(model_onnx, node)
                for node in param_prunable_nodes
            ]
        )
        four_block_sparsity_sizes = numpy.array(
            [
                get_node_num_four_block_zeros_and_size(model_onnx, node)
                for node in param_prunable_nodes
            ]
        )

        num_sparse_parameters, num_parameters = numpy.sum(sparsity_sizes, axis=0)
        num_sparse_four_blocks, num_four_blocks = numpy.sum(
            four_block_sparsity_sizes, axis=0
        )

        layer_counts, op_counts = get_layer_and_op_counts(model_onnx)
        node_shapes = extract_node_shapes(model_onnx)
        num_ops = sum(
            [
                get_num_operations(model_onnx, node, node_shapes=node_shapes)
                for node in model_onnx.graph.node
            ]
        )
        num_sparse_layers = sum(
            [is_sparse_layer(model_onnx, node) for node in model_onnx.graph.node]
        )
        num_quantized_layers = sum(
            [is_quantized_layer(model_onnx, node) for node in model_onnx.graph.node]
        )
        average_sparsity = num_sparse_parameters / num_parameters
        average_four_block_sparsity = num_sparse_four_blocks / num_four_blocks
        nodes = cls.analyze_nodes(model_onnx)

        return cls(
            layer_counts=layer_counts,
            operation_counts=op_counts,
            num_ops=num_ops,
            num_sparse_layers=num_sparse_layers,
            num_quantized_layers=num_quantized_layers,
            num_parameters=num_parameters,
            num_sparse_parameters=num_sparse_parameters,
            num_four_blocks=num_four_blocks,
            num_sparse_four_blocks=num_sparse_four_blocks,
            average_sparsity=average_sparsity,
            average_four_block_sparsity=average_four_block_sparsity,
            layers=nodes,
        )

    @staticmethod
    def analyze_nodes(model_onnx: ModelProto) -> List[NodeAnalysis]:
        """
        :param: model that contains the nodes to be analyzed
        :return: List of node analyses from model graph
        """
        node_shapes = extract_node_shapes(model_onnx)

        nodes = []
        for node in model_onnx.graph.node:
            num_ops = get_num_operations(model_onnx, node, node_shapes=node_shapes)
            node_zero_point = get_zero_point(model_onnx, node)
            num_sparse_values, node_num_values = get_node_num_zeros_and_size(
                model_onnx, node
            )
            (
                num_sparse_four_blocks,
                node_num_blocks,
            ) = get_node_num_four_block_zeros_and_size(model_onnx, node)
            node_weight = get_node_weight(model_onnx, node)
            node_analysis = NodeAnalysis(
                name=node.name,
                num_ops=num_ops,
                sparsity=get_node_sparsity(model_onnx, node),
                four_block_sparsity=get_node_four_block_sparsity(model_onnx, node),
                param_size=node_num_values,
                num_sparse_values=num_sparse_values,
                num_four_blocks=node_num_blocks,
                num_sparse_four_blocks=num_sparse_four_blocks,
                quantized_layer=is_quantized_layer(model_onnx, node),
                zero_point=node_zero_point,
                dtype=str(node_weight.dtype) if node_weight is not None else None,
            )
            nodes.append(node_analysis)

        return nodes
