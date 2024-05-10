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
"""
A module that contains schema definitions for benchmarking and/or performance
analysis results
"""

import copy
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy
import yaml
from onnx import ModelProto, NodeProto
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from sparsezoo import Model
from sparsezoo.analyze_v1.utils.helpers import numpy_array_representer
from sparsezoo.analyze_v1.utils.models import (
    DenseSparseOps,
    Entry,
    ModelEntry,
    NamedEntry,
    NodeCounts,
    NodeIO,
    NodeTimingEntry,
    OperationSummary,
    OpsSummary,
    ParameterComponent,
    ParameterSummary,
    PerformanceEntry,
    Section,
    SizedModelEntry,
    TypedEntry,
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
    load_model,
)


__all__ = [
    "NodeInferenceResult",
    "ImposedSparsificationInfo",
    "BenchmarkScenario",
    "BenchmarkResult",
    "NodeAnalysis",
    "ModelAnalysis",
]

_LOGGER = logging.getLogger()
TARGETED_LINEAR_OP_TYPES = {
    "Conv",
    "ConvInteger",
    "ConvTranspose",
    "DeformConv",
    "QLinearConv",
    "MatMul",
    "MatMulInteger",
    "QLinearMatMul",
    "Gemm",
}

# add numpy array representer to yaml
yaml.add_representer(numpy.ndarray, numpy_array_representer)


class YAMLSerializableBaseModel(BaseModel):
    """
    A BaseModel that adds a .yaml(...) function to all child classes
    """

    model_config = ConfigDict(protected_namespaces=())

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        # alias for model_dump for pydantic v2 upgrade
        # to allow for easier migration
        return self.model_dump(*args, **kwargs)

    def yaml(self, file_path: Optional[str] = None) -> Union[str, None]:
        """
        :param file_path: optional file path to save yaml to
        :return: if file_path is not given, the state of the analysis model
            as a yaml string, otherwise None
        """
        file_stream = None if file_path is None else open(file_path, "w")
        ret = yaml.dump(
            self.model_dump(), stream=file_stream, allow_unicode=True, sort_keys=False
        )

        if file_stream is not None:
            file_stream.close()

        return ret

    @classmethod
    def parse_yaml_file(cls, file_path: str):
        """
        :param file_path: path to yaml file containing model analysis data
        :return: instance of ModelAnalysis class
        """
        with open(file_path, "r") as file:
            dict_obj = yaml.safe_load(file)
        return cls.model_validate(dict_obj)

    @classmethod
    def parse_yaml_raw(cls, yaml_raw: str):
        """
        :param yaml_raw: string containing model analysis data
        :return: instance of ModelAnalysis class
        """
        dict_obj = yaml.safe_load(yaml_raw)  # unsafe: needs to load numpy
        return cls.model_validate(dict_obj)


@dataclass
class _SparseItemCount:
    name: str
    total: int = 0
    pruned: int = 0
    dense: int = 0
    quantized: int = 0


class NodeInferenceResult(YAMLSerializableBaseModel):
    """
    Schema representing node level information from a benchmarking experiment
    """

    name: str = Field(description="The node's name")
    avg_run_time: PositiveFloat = Field(
        description="Average run time for current node in milli-secs",
    )
    extras: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra arguments for DeepSparse specific results",
    )


class ImposedSparsificationInfo(YAMLSerializableBaseModel):
    """
    Schema definition for applied sparsification techniques
    """

    sparsity: Optional[PositiveFloat] = Field(
        default=None,
        description="Globally imposed sparsity level, should be " "within (0, 1.0]",
    )

    sparsity_block_structure: Optional[str] = Field(
        default=None,
        description="The sparsity block structure applied to the onnx model;"
        " ex 2:4, 4",
    )

    quantization: bool = Field(
        default=False,
        description="Flag to ascertain if quantization should be applied or not",
    )

    recipe: Optional[str] = Field(
        default=None,
        description="The recipe to be applied",
    )


class BenchmarkScenario(YAMLSerializableBaseModel):
    """
    Schema representing information for a benchmarking experiment
    """

    batch_size: PositiveInt = Field(
        description="The batch size to use for benchmarking",
    )

    num_cores: Optional[int] = Field(
        None,
        description="The number of cores to use for benchmarking, can also take "
        "in a `None` value, which represents all cores",
    )

    engine: str = Field(
        default="deepsparse",
        description="The engine to use for benchmarking, can be `deepsparse`"
        "or `onnxruntime`; defaults to `deepsparse`",
    )

    scenario: str = Field(
        default="sync",
        description="The scenario to use for benchmarking, could be `sync` or "
        "`async`; defaults to `sync`",
    )

    num_streams: Optional[int] = Field(
        default=None, description="Number of streams to use for benchmarking"
    )

    duration: int = Field(
        default=10,
        description="Number of seconds/steps the benchmark should run for, will use "
        "steps instead of seconds if `duration_in_steps` is `True`; "
        "defaults to 10",
    )

    warmup_duration: int = Field(
        default=10,
        description="Number of seconds/steps the benchmark warmup should run for, "
        "will use steps instead of seconds if `duration_in_steps` is "
        "`True`; defaults to 10 secs or steps",
    )

    instructions: Optional[str] = Field(
        default=None,
        description="Max supported instruction set available during benchmark",
    )

    analysis_only: bool = Field(
        default=False, description="Flag to only run analysis; defaults is `False`"
    )


class BenchmarkResult(YAMLSerializableBaseModel):
    """
    Schema representing results from a benchmarking experiment
    """

    setup: BenchmarkScenario = Field(
        description="Information regarding hardware, cores, batch_size, scenario and "
        "other info needed to run benchmark"
    )

    imposed_sparsification: Optional[ImposedSparsificationInfo] = Field(
        default=None,
        description="Information on sparsification techniques used for benchmarking "
        "if any",
    )

    items_per_second: float = Field(
        default=0.0, description="Number of items processed per second"
    )

    average_latency: float = Field(
        default=float("inf"), description="Average time taken per item in milli-seconds"
    )

    node_timings: Optional[List[NodeInferenceResult]] = Field(
        default=None,
        description="Node level inference results",
    )

    supported_graph_percentage: Optional[float] = Field(
        default=None,
        description="Percentage of model graph supported by the runtime engine",
    )


class NodeAnalysis(YAMLSerializableBaseModel):
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
    zero_point: Union[int, numpy.ndarray] = Field(
        description="Node zero point for quantization, default zero"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class CountSummary(BaseModel):
    items: List[_SparseItemCount]
    _precision: int = 2

    @property
    def total(self):
        return sum(item.total for item in self.items)

    @property
    def sparse_count(self):
        return sum(item.pruned for item in self.items)

    @property
    def dense_count(self):
        return sum(item.dense for item in self.items)

    @property
    def quant_count(self):
        return sum(item.quantized for item in self.items)

    @property
    def sparsity_percent(self):
        return round(self.sparse_count * 100.0 / max(1, self.total), self._precision)

    @property
    def quantized_percent(self):
        return round(self.quant_count * 100.0 / max(1, self.total), self._precision)

    @property
    def size(self):
        """
        :return: size in bits ignoring zeros
        """
        return (self.quant_count * 8 + self.dense_count * 32) * (
            1 - self.sparsity_percent / 100.0
        )

    def __add__(self, other):
        new_items = self.items + other.items
        return self.__class__(items=new_items)


class ModelAnalysisSummary(Entry, YAMLSerializableBaseModel):
    sections: List[Section]

    def pretty_print(self):
        """
        Convenience function to pretty print ModelAnalysisSummary(...) objects
        """

        for section in self.sections:
            section.pretty_print()

    @classmethod
    def from_model_analysis(
        cls,
        analysis: "ModelAnalysis",
        by_types: bool = False,
        by_layers: bool = False,
        **kwargs,
    ) -> "ModelAnalysisSummary":
        """
        Factory method to generate a ModelAnalysisSummary object from a
        sparsezoo.ModelAnalysis object

        :param analysis: The ModelAnalysis object which the newly created
            ModelAnalysisSummary object will summarize
        :param by_types: flag to summarize analysis information by param and
            op type
        :param by_layers: flag to summarize analysis information by layers
        :returns: A ModelAnalysisSummary that summarizes current analysis based
            on specified arguments
        """
        sections = []

        if by_layers:
            _LOGGER.info("Running analysis `by_layers`")
            by_layers_entries = []
            for node in analysis.nodes:
                if node.op_type not in TARGETED_LINEAR_OP_TYPES:
                    # do not include in layer-wise analysis
                    continue
                precision_dict = node.parameter_summary.precision
                dense = 0
                quantized = 0
                for precision, counts in precision_dict.items():
                    if "32" in precision:
                        # include float32, int32
                        dense += counts.total
                    else:
                        # TODO: Add support for different precisions
                        quantized += counts.total

                node_count_summary = CountSummary(
                    items=[
                        _SparseItemCount(
                            name=node.name,
                            total=node.parameter_summary.total,
                            pruned=node.parameter_summary.pruned,
                            dense=dense,
                            quantized=quantized,
                        )
                    ]
                )
                entry = NamedEntry(
                    name=node.name,
                    total=node_count_summary.total,
                    size=node_count_summary.size,
                    sparsity=node_count_summary.sparsity_percent,
                    quantized=node_count_summary.quantized_percent,
                )
                by_layers_entries.append(entry)
            if by_layers_entries:
                sections.append(
                    Section(
                        section_name="Analysis by Layers",
                        entries=by_layers_entries,
                    )
                )

        # Add Param analysis section
        param_count_summary: CountSummary = _get_param_count_summary(analysis=analysis)
        param_section = Section(
            section_name="Params",
            entries=[
                SizedModelEntry(
                    model=analysis.model_name,
                    count=param_count_summary.total,
                    size=param_count_summary.size,
                    sparsity=param_count_summary.sparsity_percent,
                    quantized=param_count_summary.quantized_percent,
                ),
            ],
        )

        # Add Ops analysis section
        ops_count_summary: CountSummary = _get_ops_count_summary(analysis=analysis)
        ops_section = Section(
            section_name="Ops",
            entries=[
                SizedModelEntry(
                    model=analysis.model_name,
                    count=ops_count_summary.total,
                    size=ops_count_summary.size,
                    sparsity=ops_count_summary.sparsity_percent,
                    quantized=ops_count_summary.quantized_percent,
                ),
            ],
        )
        if by_types:
            _LOGGER.info("Running analysis `by_types`")

            entries = []
            for item in param_count_summary.items:
                item_count_summary = CountSummary(items=[item])
                entry = TypedEntry(
                    type=item.name,
                    size=item_count_summary.size,
                    sparsity=item_count_summary.sparsity_percent,
                    quantized=item_count_summary.quantized_percent,
                )
                entries.append(entry)

            entries.append(
                TypedEntry(
                    type="Total",
                    size=param_count_summary.size,
                    sparsity=param_count_summary.sparsity_percent,
                    quantized=param_count_summary.quantized_percent,
                )
            )

            type_param_section = Section(
                section_name="Parameters by types",
                entries=entries,
            )

            sections.append(type_param_section)
            entries = []
            for item in ops_count_summary.items:
                item_count_summary = CountSummary(items=[item])
                entry = TypedEntry(
                    type=item.name,
                    size=item_count_summary.size,
                    sparsity=item_count_summary.sparsity_percent,
                    quantized=item_count_summary.quantized_percent,
                )
                entries.append(entry)

            entries.append(
                TypedEntry(
                    type="Total",
                    size=ops_count_summary.size,
                    sparsity=ops_count_summary.sparsity_percent,
                    quantized=ops_count_summary.quantized_percent,
                )
            )
            type_ops_section = Section(section_name="Ops by types", entries=entries)

            sections.append(type_ops_section)

        # Add Overall model analysis section
        overall_count_summary: CountSummary = param_count_summary + ops_count_summary
        if not analysis.benchmark_results:

            overall_section = Section(
                section_name="Overall",
                entries=[
                    ModelEntry(
                        model=analysis.model_name,
                        sparsity=overall_count_summary.sparsity_percent,
                        quantized=overall_count_summary.quantized_percent,
                    )
                ],
            )

        else:
            overall_section = Section(
                section_name="Overall",
                entries=[
                    PerformanceEntry(
                        model=analysis.model_name,
                        sparsity=overall_count_summary.sparsity_percent,
                        quantized=overall_count_summary.quantized_percent,
                        latency=benchmark_result.average_latency,
                        throughput=benchmark_result.items_per_second,
                        supported_graph=(
                            benchmark_result.supported_graph_percentage or 0.0
                        ),
                    )
                    for idx, benchmark_result in enumerate(analysis.benchmark_results)
                ],
            )

            for idx, benchmark_result in enumerate(analysis.benchmark_results):
                node_timing_section = Section(
                    section_name=f"Node Timings for Benchmark # {idx+1}",
                    entries=[
                        NodeTimingEntry(
                            node_name=node_timing.name,
                            avg_runtime=node_timing.avg_run_time,
                        )
                        for node_timing in benchmark_result.node_timings
                    ],
                )
                sections.append(node_timing_section)

        sections.extend([param_section, ops_section, overall_section])
        return cls(sections=sections)


class ModelAnalysis(YAMLSerializableBaseModel):
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

    model_name: str = Field(
        description="Optional model name, defaults to empty string",
        default="",
    )

    nodes: List[NodeAnalysis] = Field(
        description="List of analyses for each node in the model graph", default=[]
    )

    benchmark_results: List[BenchmarkResult] = Field(
        default=[],
        description="A list of different benchmarking results for the onnx model",
    )

    @classmethod
    def from_onnx(cls, onnx_file_path: Union[str, ModelProto]):
        """
        Model Analysis

        :param cls: class being constructed
        :param onnx_file_path: path to onnx file, or a loaded onnx ModelProto to
            analyze
        :return: instance of cls
        """
        path = None
        if isinstance(onnx_file_path, ModelProto):
            model_onnx = onnx_file_path
            model_name = ""
        else:
            # initially do not load the external data, if present
            # as not required for node analysis
            model_onnx = load_model(onnx_file_path, load_external_data=False)
            model_name = str(onnx_file_path)
            path = onnx_file_path

        # returns the node analysis and the model graph after loading the model with
        # external data
        node_analyses, model_graph = cls.analyze_nodes(model_onnx, path=path)

        layer_counts, op_counts = get_layer_and_op_counts(model_graph)
        layer_counts.update(op_counts)
        node_counts = layer_counts.copy()

        # these could be done with node-wise computation rather than
        # feature-wise to reduce run time

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
            model_name=model_name,
            node_counts=node_counts,
            all_nodes=all_nodes,
            parameterized=parameterized,
            non_parameterized=non_parameterized,
            parameter_summary=parameter_summary,
            operation_summary=operation_summary,
            nodes=node_analyses,
        )

    @classmethod
    def create(cls, file_path: Union[str, ModelProto]) -> "ModelAnalysis":
        """
        Factory method to create a model analysis object from an onnx filepath,
        sparsezoo stub, deployment directory, or a yaml file/raw string representing
        a `ModelAnalysis` object

        :param file_path: An instantiated ModelProto object, or path to an onnx
            model, or SparseZoo stub, or path to a deployment directory, or path
            to a yaml file or a raw yaml string representing a `ModelAnalysis`
            object. This is used to create a new ModelAnalysis object
        :returns: The created ModelAnalysis object
        """
        if not isinstance(file_path, (str, ModelProto, Path)):
            raise ValueError(
                f"Invalid file_path type {type(file_path)} passed to "
                f"ModelAnalysis.create(...)"
            )

        if isinstance(file_path, ModelProto):
            result = ModelAnalysis.from_onnx(onnx_file_path=file_path)

        elif Path(file_path).is_file():
            result = (
                ModelAnalysis.parse_yaml_file(file_path=file_path)
                if Path(file_path).suffix == ".yaml"
                else ModelAnalysis.from_onnx(onnx_file_path=file_path)
            )
        elif Path(file_path).is_dir():
            _LOGGER.info(f"Loading `model.onnx` from deployment directory {file_path}")
            result = ModelAnalysis.from_onnx(Path(file_path) / "model.onnx")

        elif file_path.startswith("zoo:"):
            # download and extract deployment directory
            Model(file_path).deployment.path
            result = ModelAnalysis.from_onnx(
                Model(file_path).deployment.get_file("model.onnx").path
            )

        elif isinstance(file_path, str):
            result = ModelAnalysis.parse_yaml_raw(yaml_raw=file_path)
        else:
            raise ValueError(
                f"Invalid argument file_path {file_path} to create ModelAnalysis"
            )

        result.model_name = file_path
        return result

    def summary(self, **kwargs) -> ModelAnalysisSummary:
        """
        :return: A ModelAnalysisSummary object that represents summary of
            current analyses
        """
        return ModelAnalysisSummary.from_model_analysis(analysis=self, **kwargs)

    def pretty_print_summary(self):
        """
        Pretty print analysis summary
        Note: pretty_print_summary() method will be deprecated from
        `ModelAnalysis` class in a future version, use
        `self.summary(...).pretty_print()` instead, where `self` is an object
        of `ModelAnalysis` class
        """
        warnings.warn(
            "my_regrettable_function will be retired in version 1.0, please "
            "use my_awesome_function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import pandas

        summary = self.summary()
        summary_copy = copy.copy(summary)
        footer = summary_copy.pop("Summary")

        # relies on pandas for pretty printing as of now
        for section_name, section_dict in summary_copy.items():
            print(f"{section_name.upper()}:")
            print(pandas.DataFrame(section_dict).T.to_string(), end="\n\n")

        print("SUMMARY:")
        for footer_key, footer_value in footer.items():
            print(f"{footer_key}: {footer_value}")

    @staticmethod
    def analyze_nodes(
        model: ModelProto, path: Optional[str] = None
    ) -> Tuple[List[NodeAnalysis], ONNXGraph]:
        """
        :param: model that contains the nodes to be analyzed
        :return: list of node analyses from model graph and ONNXGraph of loaded model
        """
        node_shapes, node_dtypes = extract_node_shapes_and_dtypes(model, path)

        if path:
            model = load_model(path)

        model_graph = ONNXGraph(model)

        nodes = []
        for node in model_graph.nodes:
            node_id = extract_node_id(node)
            node_shape = node_shapes.get(node_id)
            node_dtype = node_dtypes.get(node_id)
            node_analysis = NodeAnalysis.from_node(
                model_graph, node, node_shape=node_shape, node_dtype=node_dtype
            )
            nodes.append(node_analysis)

        return nodes, model_graph


def _get_param_count_summary(analysis: ModelAnalysis) -> CountSummary:
    alias_to_item_count: Dict[str, _SparseItemCount] = {
        name.lower(): _SparseItemCount(name=name) for name in ["Weight", "Bias"]
    }

    for node in analysis.nodes:
        for parameter in node.parameters:
            item_count = alias_to_item_count[parameter.alias]
            parameter_summary = parameter.parameter_summary

            item_count.total += parameter_summary.total
            item_count.pruned += parameter_summary.pruned

            for precision, count in parameter_summary.precision.items():
                if "32" in precision:
                    # fp32 and int32
                    item_count.dense += count.total
                else:
                    # TODO: Add support for other precisions
                    item_count.quantized += count.total

    return CountSummary(items=list(alias_to_item_count.values()))


def _get_ops_count_summary(analysis: ModelAnalysis) -> CountSummary:
    # ops summary

    pruned_param_counts = defaultdict(int)
    dense_param_counts = defaultdict(int)
    total_param_counts = defaultdict(int)
    parameterized_op_types = set()

    sparse_node_counts = defaultdict(int)
    dense_node_counts = defaultdict(int)
    total_node_counts = defaultdict(int)
    non_parameterized_op_types = set()

    for node in analysis.nodes:
        if node.parameterized_prunable:
            parameterized_op_types.add(node.op_type)
            pruned_param_counts[node.op_type] += node.parameter_summary.pruned
            total_param_counts[node.op_type] += node.parameter_summary.total

            if "float32" in node.parameter_summary.precision:
                dense_param_counts[node.op_type] += (
                    node.parameter_summary.precision["float32"].zero
                    + node.parameter_summary.precision["float32"].non_zero
                )

        else:
            non_parameterized_op_types.add(node.op_type)
            total_node_counts[node.op_type] += 1
            if node.sparse_node:
                sparse_node_counts[node.op_type] += 1
            if not node.quantized_node:
                dense_node_counts[node.op_type] += 1

    parameterized_item_counts = [
        _SparseItemCount(
            name=op_type,
            total=total_param_counts[op_type],
            pruned=pruned_param_counts[op_type],
            dense=dense_param_counts[op_type],
            quantized=total_param_counts[op_type] - dense_param_counts[op_type],
        )
        for op_type in parameterized_op_types
    ]

    non_parameterized_item_counts = [
        _SparseItemCount(
            name=op_type,
            total=total_node_counts[op_type],
            pruned=sparse_node_counts[op_type],
            dense=dense_node_counts[op_type],
            quantized=total_node_counts[op_type] - dense_node_counts[op_type],
        )
        for op_type in non_parameterized_op_types
    ]

    return CountSummary(
        items=[*parameterized_item_counts, *non_parameterized_item_counts]
    )
