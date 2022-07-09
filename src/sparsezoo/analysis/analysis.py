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

from sparsezoo.analysis.utils.models import (
    BiasAnalysis,
    DenseSparseOps,
    DenseSparseValues,
    ModelOperations,
    NodeIO,
    Operations,
    Parameters,
    WeightAnalysis,
)
from sparsezoo.utils import (
    NodeDataType,
    NodeShape,
    extract_node_id,
    extract_node_shapes_and_dtypes,
    get_layer_and_op_counts,
    get_node_bias,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_weight,
    get_node_weight_name,
    get_node_weight_shape,
    get_ops_dict,
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

    name: str = Field(description="The node's name")
    node_id: str = Field(description="The node's id (the name of its first output)")
    op_type: str = Field(description="The node's op type")

    inputs: List[NodeIO] = Field(description="The node's inputs in the onnx graph")
    outputs: List[NodeIO] = Field(description="The node's outputs in the onnx graph")

    parameters: Optional[Parameters] = Field(description="TODO")
    operations: Operations = Field(description="TODO")

    weight: Optional[WeightAnalysis] = Field(
        description="An analysis of the node's weight"
    )
    bias: Optional[BiasAnalysis] = Field(description="TODO")

    parameterized_prunable: bool = Field(
        description="Does the node have a parameterized and prunable weight"
    )
    sparse_layer: bool = Field(description="Does the node have sparse weights")
    quantized_layer: bool = Field(description="Does the node have quantized weights")
    zero_point: int = Field(
        description="Node zero point for quantization, default zero"
    )

    @classmethod
    def from_node(
        cls,
        model_onnx: ModelProto,
        node: NodeProto,
        node_shapes: Optional[Dict[str, NodeShape]] = None,
        node_dtypes: Optional[Dict[str, NodeDataType]] = None,
    ):
        """
        Node Analysis

        :param cls: class being constructed
        :param model_onnx: model onnx that node belongs to
        :param node: node being analyzed
        :param node_shapes: optional dictionary of node shapes. If not supplied,
        node_shapes will be computed
        :return: instance of cls
        """
        node_id = extract_node_id(node)
        node_shape = node_shapes.get(node_id)
        node_dtype = node_dtypes.get(node_id)

        has_input_shapes = (
            node_shape is not None and node_shape.input_shapes is not None
        )
        has_input_dtypes = (
            node_dtype is not None and node_dtype.input_dtypes is not None
        )
        has_output_shapes = (
            node_shape is not None and node_shape.output_shapes is not None
        )
        has_output_dtypes = (
            node_dtype is not None and node_dtype.output_dtypes is not None
        )

        inputs = (
            [
                NodeIO(name=name, shape=shape, dtype=str(dtype))
                for name, shape, dtype in zip(
                    node.input, node_shape.input_shapes, node_dtype.input_dtypes
                )
            ]
            if has_input_shapes and has_input_dtypes
            else []
        )
        outputs = (
            [
                NodeIO(name=name, shape=shape, dtype=str(dtype))
                for name, shape, dtype in zip(
                    node.output, node_shape.output_shapes, node_dtype.output_dtypes
                )
            ]
            if has_output_shapes and has_output_dtypes
            else []
        )

        quantized_layer = is_quantized_layer(model_onnx, node)
        ops_dict = get_ops_dict(model_onnx, node, node_shapes=node_shapes)

        node_weight = get_node_weight(model_onnx, node)
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
                parameters=Parameters(
                    single=DenseSparseValues(
                        num_non_zero=num_parameters - num_sparse_parameters,
                        num_zero=num_sparse_parameters,
                    ),
                    four_block=DenseSparseValues(
                        num_non_zero=num_four_blocks - num_sparse_four_blocks,
                        num_zero=num_sparse_four_blocks,
                    ),
                ),
                operations=Operations(
                    num_operations=DenseSparseOps(
                        dense=ops_dict["weight"]["num_dense_ops"],
                        sparse=ops_dict["weight"]["num_sparse_ops"],
                    ),
                    multiply_accumulates=DenseSparseOps(
                        dense=0,
                        sparse=0,
                    ),
                ),
                dtype=str(node_weight.dtype),
            )
            if node_weight is not None
            else None
        )

        node_bias = get_node_bias(model_onnx, node)
        bias_analysis = (
            BiasAnalysis(
                shape=list(node_bias.shape),
                operations=Operations(
                    num_operations=DenseSparseOps(
                        dense=ops_dict["bias"]["num_dense_ops"],
                        sparse=ops_dict["bias"]["num_sparse_ops"],
                    ),
                    multiply_accumulates=DenseSparseOps(
                        dense=0,
                        sparse=0,
                    ),
                ),
                dtype=str(node_bias.dtype),
            )
            if node_bias is not None
            else None
        )

        operations = Operations(
            num_operations=DenseSparseOps(
                dense=sum([ops_dict[k]["num_dense_ops"] for k in ops_dict.keys()]),
                sparse=sum([ops_dict[k]["num_sparse_ops"] for k in ops_dict.keys()]),
            ),
            multiply_accumulates=DenseSparseOps(
                dense=ops_dict["weight"]["num_dense_ops"] // 2,
                sparse=ops_dict["weight"]["num_sparse_ops"] // 2,
            ),
        )

        return cls(
            name=node.name,
            node_id=node_id,
            op_type=node.op_type,
            inputs=inputs,
            outputs=outputs,
            parameters=weight_analysis.parameters if weight_analysis else None,
            operations=operations,
            weight=weight_analysis,
            bias=bias_analysis,
            parameterized_prunable=is_parameterized_prunable_layer(model_onnx, node),
            sparse_layer=is_sparse_layer(model_onnx, node),
            quantized_layer=quantized_layer,
            zero_point=get_zero_point(model_onnx, node),
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

    total_prunable_parameters: Parameters = Field(description="TODO")
    total_operations: ModelOperations = Field(description="TODO")

    num_sparse_nodes: int = Field(description="Number of nodes with any sparsity")
    num_quantized_nodes: int = Field(description="Number of quantized nodes")

    nodes: List[NodeAnalysis] = Field(
        description="List of analyses for each layer in the model graph", default=[]
    )

    @classmethod
    def from_onnx(cls, onnx_file_path: str):
        """
        Model Analysis

        :param cls: class being constructed
        :param onnx_file_path: path to onnx file being analyzed
        :return: instance of cls
        """
        model_onnx = onnx.load(onnx_file_path)

        node_analyses = cls.analyze_nodes(model_onnx)

        layer_counts, op_counts = get_layer_and_op_counts(model_onnx)

        total_prunable_parameters = Parameters(
            single=DenseSparseValues(
                num_non_zero=sum(
                    [
                        node_analysis.weight.parameters.single.num_non_zero
                        for node_analysis in node_analyses
                        if node_analysis.parameterized_prunable
                    ]
                ),
                num_zero=sum(
                    [
                        node_analysis.weight.parameters.single.num_zero
                        for node_analysis in node_analyses
                        if node_analysis.parameterized_prunable
                    ]
                ),
            ),
            four_block=DenseSparseValues(
                num_non_zero=sum(
                    [
                        node_analysis.weight.parameters.four_block.num_non_zero
                        for node_analysis in node_analyses
                        if node_analysis.parameterized_prunable
                    ]
                ),
                num_zero=sum(
                    [
                        node_analysis.weight.parameters.four_block.num_zero
                        for node_analysis in node_analyses
                        if node_analysis.parameterized_prunable
                    ]
                ),
            ),
        )
        total_operations = ModelOperations(
            floating_or_quantized_ops=DenseSparseOps(
                dense=sum(
                    [
                        node_analysis.operations.num_operations.dense
                        for node_analysis in node_analyses
                    ]
                ),
                sparse=sum(
                    [
                        node_analysis.operations.num_operations.sparse
                        for node_analysis in node_analyses
                    ]
                ),
            ),
            floating_point_ops=DenseSparseOps(
                dense=sum(
                    [
                        node_analysis.operations.num_operations.dense
                        for node_analysis in node_analyses
                        if not node_analysis.quantized_layer
                    ]
                ),
                sparse=sum(
                    [
                        node_analysis.operations.num_operations.sparse
                        for node_analysis in node_analyses
                        if not node_analysis.quantized_layer
                    ]
                ),
            ),
            quantized_ops=DenseSparseOps(
                dense=sum(
                    [
                        node_analysis.operations.num_operations.dense
                        for node_analysis in node_analyses
                        if node_analysis.quantized_layer
                    ]
                ),
                sparse=sum(
                    [
                        node_analysis.operations.num_operations.sparse
                        for node_analysis in node_analyses
                        if node_analysis.quantized_layer
                    ]
                ),
            ),
            multiply_accumulates=DenseSparseOps(
                dense=sum(
                    [
                        node_analysis.operations.multiply_accumulates.dense
                        for node_analysis in node_analyses
                    ]
                ),
                sparse=sum(
                    [
                        node_analysis.operations.multiply_accumulates.sparse
                        for node_analysis in node_analyses
                    ]
                ),
            ),
        )

        num_sparse_nodes = len(
            [None for node_analysis in node_analyses if node_analysis.sparse_layer]
        )
        num_quantized_nodes = len(
            [None for node_analysis in node_analyses if node_analysis.quantized_layer]
        )

        return cls(
            layer_counts=layer_counts,
            non_parameterized_operator_counts=op_counts,
            total_prunable_parameters=total_prunable_parameters,
            total_operations=total_operations,
            num_sparse_nodes=num_sparse_nodes,
            num_quantized_nodes=num_quantized_nodes,
            nodes=node_analyses,
        )

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

    @staticmethod
    def analyze_nodes(model_onnx: ModelProto) -> List[NodeAnalysis]:
        """
        :param: model that contains the nodes to be analyzed
        :return: list of node analyses from model graph
        """
        node_shapes, node_dtypes = extract_node_shapes_and_dtypes(model_onnx)

        nodes = []
        for node in model_onnx.graph.node:
            node_analysis = NodeAnalysis.from_node(
                model_onnx, node, node_shapes=node_shapes, node_dtypes=node_dtypes
            )
            nodes.append(node_analysis)

        return nodes

    def yaml(self, file_path: Optional[str] = None) -> Union[str, None]:
        """
        :param file_path: optional file path to save yaml to
        :return: if file_path is not given, the state of the analysis model
            as a yaml string, otherwise None
        """
        file_stream = None if file_path is None else open(file_path, "w")
        ret = yaml.dump(
            self.dict(), stream=file_stream, allow_unicode=True, sort_keys=False
        )

        if file_stream is not None:
            file_stream.close()

        return ret
