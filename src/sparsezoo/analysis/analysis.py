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

import copy
from typing import Dict, List, Optional, Union

import numpy
import onnx
import yaml
from onnx import NodeProto
from pydantic import BaseModel, Field

from sparsezoo.analysis.utils.models import (
    DenseSparseOps,
    NodeCounts,
    NodeIO,
    OperationSummary,
    OpsSummary,
    ParameterComponent,
    ParameterSummary,
    ZeroNonZeroParams,
)
from sparsezoo.utils import (
    NodeDataType,
    NodeShape,
    ONNXGraph,
    extract_node_id,
    extract_node_shapes_and_dtypes,
    get_layer_and_op_counts,
    get_node_bias,
    get_node_bias_name,
    get_node_num_four_block_zeros_and_size,
    get_node_num_zeros_and_size,
    get_node_weight,
    get_node_weight_name,
    get_ops_dict,
    get_zero_point,
    is_four_block_sparse_layer,
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

    parameter_summary: ParameterSummary = Field(
        description="The node's total number of parameters (excluding bias parameters)"
    )
    operation_summary: OperationSummary = Field(
        description="The node's total number of operations (including bias ops and "
        "other op types such as max pool)"
    )

    parameters: List[ParameterComponent] = Field(
        description="The parameters belonging to the node such as weight and bias"
    )

    parameterized_prunable: bool = Field(
        description="Does the node have a parameterized and prunable weight"
    )
    sparse_node: bool = Field(description="Does the node have sparse weights")
    quantized_node: bool = Field(description="Does the node have quantized weights")
    zero_point: int = Field(
        description="Node zero point for quantization, default zero"
    )

    @classmethod
    def from_node(
        cls,
        model_graph: ONNXGraph,
        node: NodeProto,
        node_shape: Union[NodeShape, None],
        node_dtype: Union[NodeDataType, None],
    ):
        """
        Node Analysis

        :param cls: class being constructed
        :param model_graph: instance of ONNXGraph containing node
        :param node: node being analyzed
        :param node_shape: the shape of the node
        :param node_dtype: the datatype of the node
        :return: instance of cls
        """
        node_id = extract_node_id(node)

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

        sparse_node = is_sparse_layer(model_graph, node)
        quantized_node = is_quantized_layer(model_graph, node)

        node_weight = get_node_weight(model_graph, node)
        node_bias = get_node_bias(model_graph, node)
        node_bias_size = node_bias.size if node_bias is not None else 0
        param_dtypes = [
            str(param.dtype) for param in [node_weight, node_bias] if param is not None
        ]
        num_sparse_parameters, num_parameters = get_node_num_zeros_and_size(
            model_graph, node
        )
        (
            num_sparse_four_blocks,
            num_four_blocks,
        ) = get_node_num_four_block_zeros_and_size(model_graph, node)
        parameter_summary = ParameterSummary(
            total=num_parameters + node_bias_size,
            pruned=num_sparse_parameters,
            block_structure={
                "single": ZeroNonZeroParams(
                    zero=num_sparse_parameters + node_bias_size,
                    non_zero=num_parameters - num_sparse_parameters,
                ),
                "block4": ZeroNonZeroParams(
                    zero=num_sparse_four_blocks,
                    non_zero=num_four_blocks - num_sparse_four_blocks,
                ),
            },
            precision={
                dtype: ZeroNonZeroParams(
                    zero=(
                        num_sparse_parameters
                        if node_weight is not None and str(node_weight.dtype) == dtype
                        else 0
                    )
                    + (
                        node_bias_size - numpy.count_nonzero(node_bias)
                        if node_bias is not None and str(node_bias.dtype) == dtype
                        else 0
                    ),
                    non_zero=(
                        num_parameters - num_sparse_parameters
                        if node_weight is not None and str(node_weight.dtype) == dtype
                        else 0
                    )
                    + (
                        numpy.count_nonzero(node_bias)
                        if node_bias is not None and str(node_bias.dtype) == dtype
                        else 0
                    ),
                )
                for dtype in param_dtypes
            },
        )

        def _sum_across_keys(dict, key):
            return sum([dict[k][key] for k in dict.keys()])

        is_four_block_sparse = is_four_block_sparse_layer(model_graph, node)
        single_ops_dict = get_ops_dict(
            model_graph, node, node_shape=node_shape, is_four_block_sparse=False
        )
        four_block_ops_dict = get_ops_dict(
            model_graph, node, node_shape=node_shape, is_four_block_sparse=True
        )
        true_ops_dict = (
            single_ops_dict if not is_four_block_sparse else four_block_ops_dict
        )

        # collect all dtypes include "other" (non weight or bias)
        operation_dtypes = copy.deepcopy(param_dtypes)
        first_output = next(iter(outputs), None)
        other_op_dtype = first_output.dtype if first_output is not None else "unknown"
        if (
            true_ops_dict["other"]["num_dense_ops"]
            + true_ops_dict["other"]["num_sparse_ops"]
            != 0
        ):
            operation_dtypes.append(other_op_dtype)

        operation_summary = OperationSummary(
            ops=OpsSummary(
                total=_sum_across_keys(true_ops_dict, "num_dense_ops")
                + _sum_across_keys(true_ops_dict, "num_sparse_ops"),
                pruned=_sum_across_keys(true_ops_dict, "num_sparse_ops"),
                block_structure={
                    "single": DenseSparseOps(
                        dense=_sum_across_keys(single_ops_dict, "num_dense_ops"),
                        sparse=_sum_across_keys(single_ops_dict, "num_sparse_ops"),
                    ),
                    "block4": DenseSparseOps(
                        dense=_sum_across_keys(four_block_ops_dict, "num_dense_ops"),
                        sparse=_sum_across_keys(four_block_ops_dict, "num_sparse_ops"),
                    ),
                },
                precision={
                    dtype: DenseSparseOps(
                        dense=(
                            (
                                true_ops_dict["weight"]["num_dense_ops"]
                                if node_weight is not None
                                and str(node_weight.dtype) == dtype
                                else 0
                            )
                            + (
                                true_ops_dict["bias"]["num_dense_ops"]
                                if node_bias is not None
                                and str(node_bias.dtype) == dtype
                                else 0
                            )
                            + (
                                true_ops_dict["other"]["num_dense_ops"]
                                if other_op_dtype == dtype
                                else 0
                            )
                        ),
                        sparse=(
                            (
                                true_ops_dict["weight"]["num_sparse_ops"]
                                if node_weight is not None
                                and str(node_weight.dtype) == dtype
                                else 0
                            )
                            + (
                                true_ops_dict["bias"]["num_sparse_ops"]
                                if node_bias is not None
                                and str(node_bias.dtype) == dtype
                                else 0
                            )
                            + (
                                true_ops_dict["other"]["num_sparse_ops"]
                                if other_op_dtype == dtype
                                else 0
                            )
                        ),
                    )
                    for dtype in operation_dtypes
                },
            ),
            macs=OpsSummary(
                total=(
                    true_ops_dict["weight"]["num_dense_ops"]
                    + true_ops_dict["weight"]["num_sparse_ops"]
                )
                // 2,
                pruned=true_ops_dict["weight"]["num_sparse_ops"] // 2,
                block_structure={
                    "single": DenseSparseOps(
                        dense=single_ops_dict["weight"]["num_dense_ops"] // 2,
                        sparse=single_ops_dict["weight"]["num_sparse_ops"] // 2,
                    ),
                    "block4": DenseSparseOps(
                        dense=four_block_ops_dict["weight"]["num_dense_ops"] // 2,
                        sparse=four_block_ops_dict["weight"]["num_sparse_ops"] // 2,
                    ),
                },
                precision={
                    str(node_weight.dtype): DenseSparseOps(
                        dense=true_ops_dict["weight"]["num_dense_ops"] // 2,
                        sparse=true_ops_dict["weight"]["num_sparse_ops"] // 2,
                    )
                }
                if node_weight is not None
                else {},
            ),
        )

        parameters = []
        if node_weight is not None:
            parameters.append(
                ParameterComponent(
                    alias="weight",
                    name=get_node_weight_name(model_graph, node),
                    shape=node_weight.shape,
                    parameter_summary=ParameterSummary(
                        total=num_parameters,
                        pruned=num_sparse_parameters,
                        block_structure={
                            "single": ZeroNonZeroParams(
                                zero=num_sparse_parameters,
                                non_zero=num_parameters - num_sparse_parameters,
                            ),
                            "block4": ZeroNonZeroParams(
                                zero=num_sparse_four_blocks,
                                non_zero=num_four_blocks - num_sparse_four_blocks,
                            ),
                        },
                        precision={
                            str(node_weight.dtype): ZeroNonZeroParams(
                                zero=num_sparse_parameters,
                                non_zero=num_parameters - num_sparse_parameters,
                            )
                        }
                        if node_weight is not None
                        else {},
                    ),
                    dtype=str(node_weight.dtype),
                )
            )
        if node_bias is not None:
            parameters.append(
                ParameterComponent(
                    alias="bias",
                    name=get_node_bias_name(node),
                    shape=node_bias.shape,
                    parameter_summary=ParameterSummary(
                        total=node_bias_size,
                        pruned=0,
                        block_structure={
                            "single": ZeroNonZeroParams(
                                zero=node_bias_size - numpy.count_nonzero(node_bias),
                                non_zero=numpy.count_nonzero(node_bias),
                            ),
                            "block4": ZeroNonZeroParams(
                                zero=0,
                                non_zero=0,
                            ),
                        },
                        precision={
                            str(node_bias.dtype): ZeroNonZeroParams(
                                zero=node_bias_size - numpy.count_nonzero(node_bias),
                                non_zero=numpy.count_nonzero(node_bias),
                            )
                        }
                        if node_bias is not None
                        else {},
                    ),
                    dtype=str(node_bias.dtype),
                )
            )

        return cls(
            name=node.name,
            node_id=node_id,
            op_type=node.op_type,
            inputs=inputs,
            outputs=outputs,
            parameter_summary=parameter_summary,
            operation_summary=operation_summary,
            parameters=parameters,
            parameterized_prunable=is_parameterized_prunable_layer(model_graph, node),
            sparse_node=sparse_node,
            quantized_node=quantized_node,
            zero_point=get_zero_point(model_graph, node),
        )


class ModelAnalysis(BaseModel):
    """
    Pydantic model for analysis of a model
    """

    node_counts: Dict[str, int] = Field(description="The number of each node op type")
    all_nodes: NodeCounts = Field(
        description="The number of nodes grouped by their attributes"
    )
    parameterized: NodeCounts = Field(
        description="The number of nodes which are parameterized grouped by their "
        "attributes"
    )
    non_parameterized: NodeCounts = Field(
        description="The number of nodes which are not parameterized grouped by "
        "their attributes"
    )

    parameter_summary: ParameterSummary = Field(
        description="A summary of all the parameters in the model"
    )
    operation_summary: OperationSummary = Field(
        description="A summary of all the operations in the model"
    )

    nodes: List[NodeAnalysis] = Field(
        description="List of analyses for each node in the model graph", default=[]
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
        model_graph = ONNXGraph(model_onnx)

        node_analyses = cls.analyze_nodes(model_graph)

        layer_counts, op_counts = get_layer_and_op_counts(model_graph)
        layer_counts.update(op_counts)
        node_counts = layer_counts.copy()

        # these could be done with node-wise computation rather than feature-wise
        # to reduce run time

        all_nodes = NodeCounts(
            total=len(node_analyses),
            quantized=len(
                [1 for node_analysis in node_analyses if node_analysis.quantized_node]
            ),
            # quantizable
            pruned=len(
                [1 for node_analysis in node_analyses if node_analysis.sparse_node]
            ),
            prunable=len(
                [
                    1
                    for node_analysis in node_analyses
                    if node_analysis.parameterized_prunable
                ]
            ),
        )

        parameterized = NodeCounts(
            total=len(
                [
                    1
                    for node_analysis in node_analyses
                    if node_analysis.parameterized_prunable
                ]
            ),
            quantized=len(
                [
                    1
                    for node_analysis in node_analyses
                    if node_analysis.parameterized_prunable
                    and node_analysis.quantized_node
                ]
            ),
            # quantizable
            pruned=len(
                [
                    1
                    for node_analysis in node_analyses
                    if node_analysis.parameterized_prunable
                    and node_analysis.sparse_node
                ]
            ),
            prunable=len(
                [
                    1
                    for node_analysis in node_analyses
                    if node_analysis.parameterized_prunable
                ]
            ),
        )

        non_parameterized = NodeCounts(
            total=len(
                [
                    1
                    for node_analysis in node_analyses
                    if not node_analysis.parameterized_prunable
                ]
            ),
            quantized=len(
                [
                    1
                    for node_analysis in node_analyses
                    if not node_analysis.parameterized_prunable
                    and node_analysis.quantized_node
                ]
            ),
            # quantizable
            pruned=len(
                [
                    1
                    for node_analysis in node_analyses
                    if not node_analysis.parameterized_prunable
                    and node_analysis.sparse_node
                ]
            ),
            prunable=len(
                [
                    1
                    for node_analysis in node_analyses
                    if not node_analysis.parameterized_prunable
                ]
            ),
        )

        all_parameter_dtypes = []
        all_ops_operation_dtypes = []
        all_macs_operation_dtypes = []
        for node_analysis in node_analyses:
            all_parameter_dtypes.extend(
                node_analysis.parameter_summary.precision.keys()
            )
            all_ops_operation_dtypes.extend(
                node_analysis.operation_summary.ops.precision.keys()
            )
            all_macs_operation_dtypes.extend(
                node_analysis.operation_summary.macs.precision.keys()
            )
        all_ops_operation_dtypes = set(all_ops_operation_dtypes)
        all_macs_operation_dtypes = set(all_macs_operation_dtypes)
        all_parameter_dtypes = set(all_parameter_dtypes)

        # this can be done with better run time efficiency with a recursive summing algo
        parameter_summary = ParameterSummary(
            total=sum(
                [
                    node_analysis.parameter_summary.total
                    for node_analysis in node_analyses
                ]
            ),
            pruned=sum(
                [
                    node_analysis.parameter_summary.pruned
                    for node_analysis in node_analyses
                ]
            ),
            block_structure={
                "single": ZeroNonZeroParams(
                    zero=sum(
                        [
                            node_analysis.parameter_summary.block_structure[
                                "single"
                            ].zero
                            for node_analysis in node_analyses
                        ]
                    ),
                    non_zero=sum(
                        [
                            node_analysis.parameter_summary.block_structure[
                                "single"
                            ].non_zero
                            for node_analysis in node_analyses
                        ]
                    ),
                ),
                "block4": ZeroNonZeroParams(
                    zero=sum(
                        [
                            node_analysis.parameter_summary.block_structure[
                                "block4"
                            ].zero
                            for node_analysis in node_analyses
                        ]
                    ),
                    non_zero=sum(
                        [
                            node_analysis.parameter_summary.block_structure[
                                "block4"
                            ].non_zero
                            for node_analysis in node_analyses
                        ]
                    ),
                ),
            },
            precision={
                dtype: ZeroNonZeroParams(
                    zero=sum(
                        [
                            node_analysis.parameter_summary.precision[dtype].zero
                            for node_analysis in node_analyses
                            if dtype in node_analysis.parameter_summary.precision
                        ]
                    ),
                    non_zero=sum(
                        [
                            node_analysis.parameter_summary.precision[dtype].non_zero
                            for node_analysis in node_analyses
                            if dtype in node_analysis.parameter_summary.precision
                        ]
                    ),
                )
                for dtype in all_parameter_dtypes
            },
        )

        operation_summary = OperationSummary(
            ops=OpsSummary(
                total=sum(
                    [
                        node_analysis.operation_summary.ops.total
                        for node_analysis in node_analyses
                    ]
                ),
                pruned=sum(
                    [
                        node_analysis.operation_summary.ops.pruned
                        for node_analysis in node_analyses
                    ]
                ),
                block_structure={
                    "single": DenseSparseOps(
                        dense=sum(
                            [
                                node_analysis.operation_summary.ops.block_structure[
                                    "single"
                                ].dense
                                for node_analysis in node_analyses
                            ]
                        ),
                        sparse=sum(
                            [
                                node_analysis.operation_summary.ops.block_structure[
                                    "single"
                                ].sparse
                                for node_analysis in node_analyses
                            ]
                        ),
                    ),
                    "block4": DenseSparseOps(
                        dense=sum(
                            [
                                node_analysis.operation_summary.ops.block_structure[
                                    "block4"
                                ].dense
                                for node_analysis in node_analyses
                            ]
                        ),
                        sparse=sum(
                            [
                                node_analysis.operation_summary.ops.block_structure[
                                    "block4"
                                ].sparse
                                for node_analysis in node_analyses
                            ]
                        ),
                    ),
                },
                precision={
                    dtype: DenseSparseOps(
                        dense=sum(
                            [
                                node_analysis.operation_summary.ops.precision[
                                    dtype
                                ].dense
                                for node_analysis in node_analyses
                                if dtype
                                in node_analysis.operation_summary.ops.precision
                            ]
                        ),
                        sparse=sum(
                            [
                                node_analysis.operation_summary.ops.precision[
                                    dtype
                                ].sparse
                                for node_analysis in node_analyses
                                if dtype
                                in node_analysis.operation_summary.ops.precision
                            ]
                        ),
                    )
                    for dtype in all_ops_operation_dtypes
                },
            ),
            macs=OpsSummary(
                total=sum(
                    [
                        node_analysis.operation_summary.macs.total
                        for node_analysis in node_analyses
                    ]
                ),
                pruned=sum(
                    [
                        node_analysis.operation_summary.macs.pruned
                        for node_analysis in node_analyses
                    ]
                ),
                block_structure={
                    "single": DenseSparseOps(
                        dense=sum(
                            [
                                node_analysis.operation_summary.macs.block_structure[
                                    "single"
                                ].dense
                                for node_analysis in node_analyses
                            ]
                        ),
                        sparse=sum(
                            [
                                node_analysis.operation_summary.macs.block_structure[
                                    "single"
                                ].sparse
                                for node_analysis in node_analyses
                            ]
                        ),
                    ),
                    "block4": DenseSparseOps(
                        dense=sum(
                            [
                                node_analysis.operation_summary.macs.block_structure[
                                    "block4"
                                ].dense
                                for node_analysis in node_analyses
                            ]
                        ),
                        sparse=sum(
                            [
                                node_analysis.operation_summary.macs.block_structure[
                                    "block4"
                                ].sparse
                                for node_analysis in node_analyses
                            ]
                        ),
                    ),
                },
                precision={
                    dtype: DenseSparseOps(
                        dense=sum(
                            [
                                node_analysis.operation_summary.macs.precision[
                                    dtype
                                ].dense
                                for node_analysis in node_analyses
                                if dtype
                                in node_analysis.operation_summary.macs.precision
                            ]
                        ),
                        sparse=sum(
                            [
                                node_analysis.operation_summary.macs.precision[
                                    dtype
                                ].sparse
                                for node_analysis in node_analyses
                                if dtype
                                in node_analysis.operation_summary.macs.precision
                            ]
                        ),
                    )
                    for dtype in all_macs_operation_dtypes
                },
            ),
        )

        return cls(
            node_counts=node_counts,
            all_nodes=all_nodes,
            parameterized=parameterized,
            non_parameterized=non_parameterized,
            parameter_summary=parameter_summary,
            operation_summary=operation_summary,
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
    def analyze_nodes(model_graph: ONNXGraph) -> List[NodeAnalysis]:
        """
        :param: model that contains the nodes to be analyzed
        :return: list of node analyses from model graph
        """
        node_shapes, node_dtypes = extract_node_shapes_and_dtypes(model_graph.model)

        nodes = []
        for node in model_graph.nodes:
            node_id = extract_node_id(node)
            node_shape = node_shapes.get(node_id)
            node_dtype = node_dtypes.get(node_id)
            node_analysis = NodeAnalysis.from_node(
                model_graph, node, node_shape=node_shape, node_dtype=node_dtype
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
