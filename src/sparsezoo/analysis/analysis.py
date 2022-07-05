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

from typing import Dict, List, Optional, Union

import onnx
import yaml
from onnx import ModelProto, NodeProto
from pydantic import BaseModel, Field

from sparsezoo.utils import (
    NodeShape,
    extract_node_id,
    extract_node_shapes,
    get_layer_and_op_counts,
    get_node_four_block_sparsity,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_sparsity,
    get_node_weight,
    get_node_weight_name,
    get_node_weight_shape,
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


class NodeIO(BaseModel):
    """
    Pydantic model for the inputs and outputs of a node in the onnx model graph
    """

    name: str = Field(description="Name of this input/output in onnx model graph")
    shape: Optional[List[Union[None, int]]] = Field(
        description="Shape of this input/output in onnx model graph (assuming a "
        "batch size of 1)"
    )


class WeightAnalysis(BaseModel):
    """
    Pydantic model for the weight of a node
    """

    name: Optional[str] = Field(description="The name of this weight")
    shape: Optional[List[int]] = Field(
        description="This weight's shape (assuming a batch size of 1)"
    )
    sparsity: float = Field(description="Proportion of zeros in this weight")
    four_block_sparsity: float = Field(
        description="Proportion of zero blocks in this weight"
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


class NodeAnalysis(BaseModel):
    """
    Pydantic model for the analysis of a node within a model
    """

    name: str = Field(description="This node's name")
    node_id: str = Field(description="This node's id (the name of its first output)")
    op_type: str = Field(description="This node's op type")

    inputs: List[NodeIO] = Field(description="This node's inputs in the onnx graph")
    outputs: List[NodeIO] = Field(description="This node's outputs in the onnx graph")

    weight: Optional[WeightAnalysis] = Field(
        description="An analysis of this node's weight"
    )

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
    is_sparse_layer: bool = Field(description="Does this node have sparse weights")
    is_quantized_layer: bool = Field(
        description="Does this node have quantized weights"
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
        node_id = extract_node_id(node)
        node_shape = node_shapes.get(node_id)

        inputs = (
            [
                NodeIO(name=name, shape=shape)
                for name, shape in zip(node.input, node_shape.input_shapes)
            ]
            if node_shape is not None and node_shape.input_shapes is not None
            else []
        )
        outputs = (
            [
                NodeIO(name=name, shape=shape)
                for name, shape in zip(node.output, node_shape.output_shapes)
            ]
            if node_shape is not None and node_shape.output_shapes is not None
            else []
        )

        node_weight = get_node_weight(model_onnx, node)
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
        weight_analysis = (
            WeightAnalysis(
                name=get_node_weight_name(model_onnx, node),
                shape=get_node_weight_shape(model_onnx, node),
                sparsity=get_node_sparsity(model_onnx, node),
                four_block_sparsity=get_node_four_block_sparsity(model_onnx, node),
                num_parameters=num_parameters,
                num_sparse_parameters=num_sparse_parameters,
                num_four_blocks=num_four_blocks,
                num_sparse_four_blocks=num_sparse_four_blocks,
            )
            if node_weight is not None
            else None
        )

        return cls(
            name=node.name,
            node_id=node_id,
            op_type=node.op_type,
            inputs=inputs,
            outputs=outputs,
            weight=weight_analysis,
            parameterized_and_prunable=is_parameterized_prunable_layer(
                model_onnx, node
            ),
            num_dense_ops=num_dense_ops,
            num_sparse_ops=num_sparse_ops,
            is_sparse_layer=is_sparse_layer(model_onnx, node),
            is_quantized_layer=is_quantized_layer(model_onnx, node),
            zero_point=get_zero_point(model_onnx, node),
            dtype=str(node_weight.dtype) if node_weight is not None else None,
        )


class ModelAnalysis(BaseModel):
    """
    Pydantic model for analysis of a model
    """

    layer_counts: Dict[str, int] = Field(
        description="Overview of nodes with parameterized weights", default={}
    )
    non_parameterized_operator_counts: Dict[str, int] = Field(
        description="Overview of nodes without parameterized weights", default={}
    )
    num_sparse_ops: int = Field(
        description="The total number of (floating point and/or integer) operations "
        "not performed because of sparsification"
    )
    num_dense_ops: int = Field(
        description="The total number of (floating point and/or integer) operations "
        "performed in one forward pass of the model"
    )
    num_sparse_quantized_ops: int = Field(
        description="Number of sparse quantized operations (for nodes with "
        "parameterized weights)"
    )
    num_dense_quantized_ops: int = Field(
        description="Number of dense quantized operations (for nodes with "
        "parameterized weights)"
    )
    num_sparse_floating_point_ops: int = Field(
        description="Number of sparse floating point operations (for nodes with "
        "parameterized weights)"
    )
    num_dense_floating_point_ops: int = Field(
        description="Number of dense floating point operations (for nodes with "
        "parameterized weights)"
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

    nodes: List[NodeAnalysis] = Field(
        description="List of analyses for each layer in the model graph", default=[]
    )

    @classmethod
    def from_onnx(cls, onnx_file_path: str):
        """
        Class constructor for Model Analysis

        :param cls: class being constructed
        :param onnx_file_path: path to onnx file being analyzed
        :return: instance of ModelAnalysis class
        """
        model_onnx = onnx.load(onnx_file_path)

        node_analyses = cls.analyze_nodes(model_onnx)

        layer_counts, op_counts = get_layer_and_op_counts(model_onnx)

        num_parameters = sum(
            [
                node_analysis.weight.num_parameters
                for node_analysis in node_analyses
                if node_analysis.weight is not None
            ]
        )
        num_sparse_parameters = sum(
            [
                node_analysis.weight.num_sparse_parameters
                for node_analysis in node_analyses
                if node_analysis.weight is not None
            ]
        )
        num_four_blocks = sum(
            [
                node_analysis.weight.num_four_blocks
                for node_analysis in node_analyses
                if node_analysis.weight is not None
            ]
        )
        num_sparse_four_blocks = sum(
            [
                node_analysis.weight.num_sparse_four_blocks
                for node_analysis in node_analyses
                if node_analysis.weight is not None
            ]
        )

        num_sparse_quantized_ops = sum(
            [
                node_analysis.num_sparse_ops
                for node_analysis in node_analyses
                if node_analysis.dtype and "int" in node_analysis.dtype
            ]
        )
        num_dense_quantized_ops = sum(
            [
                node_analysis.num_dense_ops
                for node_analysis in node_analyses
                if node_analysis.dtype and "int" in node_analysis.dtype
            ]
        )
        num_sparse_floating_point_ops = sum(
            [
                node_analysis.num_sparse_ops
                for node_analysis in node_analyses
                if node_analysis.dtype and "float" in node_analysis.dtype
            ]
        )
        num_dense_floating_point_ops = sum(
            [
                node_analysis.num_dense_ops
                for node_analysis in node_analyses
                if node_analysis.dtype and "float" in node_analysis.dtype
            ]
        )

        return cls(
            layer_counts=layer_counts,
            non_parameterized_operator_counts=op_counts,
            num_dense_ops=sum(
                [node_analysis.num_dense_ops for node_analysis in node_analyses]
            ),
            num_sparse_ops=sum(
                [node_analysis.num_sparse_ops for node_analysis in node_analyses]
            ),
            num_sparse_layers=len(
                [
                    None
                    for node_analysis in node_analyses
                    if node_analysis.is_sparse_layer
                ]
            ),
            num_quantized_layers=len(
                [
                    None
                    for node_analysis in node_analyses
                    if node_analysis.is_quantized_layer
                ]
            ),
            num_parameters=num_parameters,
            num_sparse_parameters=num_sparse_parameters,
            num_four_blocks=num_four_blocks,
            num_sparse_four_blocks=num_sparse_four_blocks,
            num_dense_quantized_ops=num_dense_quantized_ops,
            num_sparse_quantized_ops=num_sparse_quantized_ops,
            num_sparse_floating_point_ops=num_sparse_floating_point_ops,
            num_dense_floating_point_ops=num_dense_floating_point_ops,
            average_sparsity=(num_sparse_parameters / num_parameters),
            average_four_block_sparsity=(num_sparse_four_blocks / num_four_blocks),
            nodes=node_analyses,
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

    def yaml(self, file_path: Optional[str] = None) -> Union[str, None]:
        """
        :param file_path: optional file path to save yaml to
        :return: if file_path is not given, the state of this analysis model
            as a yaml string, otherwise None
        """
        if file_path is None:
            return yaml.dump(self.dict(), allow_unicode=True, sort_keys=False)
        else:
            with open(file_path, "w") as f:
                yaml.dump(self.dict(), f, allow_unicode=True, sort_keys=False)

    @classmethod
    def parse_yaml_file(cls, file_path: str):
        """
        :param file_path: path to yaml file containing model analysis data
        :return: instance of ModelAnalysis class
        """
        with open(file_path, "r") as file:
            dict_obj = yaml.safe_load(file)
        return cls.parse_obj(dict_obj)

    @classmethod
    def parse_yaml_raw(cls, yaml_raw: str):
        """
        :param yaml_raw: string containing model analysis data
        :return: instance of ModelAnalysis class
        """
        dict_obj = yaml.safe_load(yaml_raw)  # unsafe: needs to load numpy
        return cls.parse_obj(dict_obj)
