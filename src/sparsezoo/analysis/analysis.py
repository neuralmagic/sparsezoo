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

import onnx
from onnx import ModelProto, NodeProto

from pydantic import BaseModel, Field
from sparsezoo.analysis.utils import (
    NodeShape,
    extract_node_shapes,
    get_layer_and_op_counts,
    get_node_four_block_sparsity,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_sparsity,
    get_node_weight,
    get_node_weight_name,
    get_num_dense_and_sparse_ops,
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
    op_type: str = Field(description="This node's op type")
    weight_name: Optional[str] = Field(description="The name of this node's weight")
    parameterized_and_prunable: bool = Field(
        description="Does this node have a parameterized and prunable weight"
    )
    num_dense_ops: int = Field(
        description="The number of (floating point and/or integer) operations "
        "performed in this node"
    )
    num_sparse_ops: int = Field(
        description="The number of (floating point and/or integer) operations "
        "in this node not performed because of sparsification"
    )
    sparsity: float = Field(description="Proportion of zeros in this node's parameter")
    four_block_sparsity: float = Field(
        description="Proportion of zero blocks in this node's parameter"
    )
    is_sparse_layer: bool = Field(description="Does this node have sparse weights")
    is_quantized_layer: bool = Field(
        description="Does this node have quantized weights"
    )
    num_parameters: int = Field(description="Size of this node's parameter")
    num_sparse_parameters: int = Field(
        description="Number of zeros in this node's parameter"
    )
    num_four_blocks: int = Field(
        description="Number of four blocks in this node's parameter"
    )
    num_sparse_four_blocks: int = Field(
        description="Number of four blocks with all zeros in this node's parameter"
    )
    zero_point: int = Field(
        description="Node zero point for quantization, default zero"
    )
    dtype: Optional[str] = Field(
        description="This node's parameter's data type", default=None
    )

    @classmethod
    def from_node(
        cls,
        model_onnx: ModelProto,
        node: NodeProto,
        node_shapes: Optional[Dict[str, NodeShape]] = None,
    ):
        """
        Class constructor for Node Analysis

        :param cls: class being constructed
        :param model_onnx: model onnx that node belongs to
        :param node: node being analyzed
        :param node_shapes: optional dictionary of node shapes. If not supplied,
        node_shapes will be computed
        :return: instance of NodeAnalysis class
        """
        num_dense_ops, num_sparse_ops = get_num_dense_and_sparse_ops(
            model_onnx, node, node_shapes=node_shapes
        )
        num_sparse_parameters, num_parameters = get_node_num_zeros_and_size(
            model_onnx, node
        )
        (
            num_sparse_four_blocks,
            num_four_blocks,
        ) = get_node_num_four_block_zeros_and_size(model_onnx, node)
        node_weight = get_node_weight(model_onnx, node)

        return cls(
            name=node.name,
            op_type=node.op_type,
            weight_name=get_node_weight_name(model_onnx, node),
            parameterized_and_prunable=is_parameterized_prunable_layer(
                model_onnx, node
            ),
            num_dense_ops=num_dense_ops,
            num_sparse_ops=num_sparse_ops,
            sparsity=get_node_sparsity(model_onnx, node),
            four_block_sparsity=get_node_four_block_sparsity(model_onnx, node),
            num_parameters=num_parameters,
            num_sparse_parameters=num_sparse_parameters,
            num_four_blocks=num_four_blocks,
            num_sparse_four_blocks=num_sparse_four_blocks,
            is_sparse_layer=is_sparse_layer(model_onnx, node),
            is_quantized_layer=is_quantized_layer(model_onnx, node),
            zero_point=get_zero_point(model_onnx, node),
            dtype=str(node_weight.dtype) if node_weight is not None else None,
        )


class ModelAnalysis(BaseModel):
    """
    Pydantic model for analysis of a model
    """

    layer_counts: Dict[str, int] = Field(description="Overview of layers", default={})
    non_parameterized_operator_counts: Dict[str, int] = Field(
        description="Overview of operations (nodes that are not layers)", default={}
    )
    num_dense_ops: int = Field(
        description="The total number of (floating point and/or integer) operations "
        "performed in one forward pass of the model"
    )
    num_sparse_ops: int = Field(
        description="The total number of (floating point and/or integer) operations "
        "not performed because of sparsification"
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

        layer_analyses = cls.analyze_nodes(model_onnx)

        layer_counts, op_counts = get_layer_and_op_counts(model_onnx)

        num_parameters = sum(
            [layer_analysis.num_parameters for layer_analysis in layer_analyses]
        )
        num_sparse_parameters = sum(
            [layer_analysis.num_sparse_parameters for layer_analysis in layer_analyses]
        )
        num_four_blocks = sum(
            [layer_analysis.num_four_blocks for layer_analysis in layer_analyses]
        )
        num_sparse_four_blocks = sum(
            [layer_analysis.num_sparse_four_blocks for layer_analysis in layer_analyses]
        )

        return cls(
            layer_counts=layer_counts,
            non_parameterized_operator_counts=op_counts,
            num_dense_ops=sum(
                [layer_analysis.num_dense_ops for layer_analysis in layer_analyses]
            ),
            num_sparse_ops=sum(
                [layer_analysis.num_sparse_ops for layer_analysis in layer_analyses]
            ),
            num_sparse_layers=len(
                [
                    None
                    for layer_analysis in layer_analyses
                    if layer_analysis.is_sparse_layer
                ]
            ),
            num_quantized_layers=len(
                [
                    None
                    for layer_analysis in layer_analyses
                    if layer_analysis.is_quantized_layer
                ]
            ),
            num_parameters=num_parameters,
            num_sparse_parameters=num_sparse_parameters,
            num_four_blocks=num_four_blocks,
            num_sparse_four_blocks=num_sparse_four_blocks,
            average_sparsity=(num_sparse_parameters / num_parameters),
            average_four_block_sparsity=(num_sparse_four_blocks / num_four_blocks),
            layers=layer_analyses,
        )

    @staticmethod
    def analyze_nodes(model_onnx: ModelProto) -> List[NodeAnalysis]:
        """
        :param: model that contains the nodes to be analyzed
        :return: list of node analyses from model graph
        """
        node_shapes = extract_node_shapes(model_onnx)

        nodes = []
        for node in model_onnx.graph.node:
            node_analysis = NodeAnalysis.from_node(
                model_onnx, node, node_shapes=node_shapes
            )
            nodes.append(node_analysis)

        return nodes
