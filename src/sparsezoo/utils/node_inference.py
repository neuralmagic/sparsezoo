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
NOTE: Adapted from sparseml/onnx/utils/helpers.py
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy
import onnx
from onnx import ModelProto
from onnx.helper import make_empty_tensor_value_info
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from sparsezoo.utils.onnx import extract_node_id


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "NodeShape",
    "NodeDataType",
    "extract_node_shapes_and_dtypes",
]


"""
Tuple containing a node id and its input and output shapes
"""
NodeShape = NamedTuple(
    "NodeShape",
    [
        ("id", str),
        ("input_shapes", Union[List[List[int]], None]),
        ("output_shapes", Union[List[List[int]], None]),
    ],
)
NodeDataType = NamedTuple(
    "NodeDataType",
    [
        ("id", str),
        ("input_dtypes", Union[List[numpy.dtype], None]),
        ("output_dtypes", Union[List[numpy.dtype], None]),
    ],
)


def extract_nodes_shapes_and_dtypes_ort(
    model: ModelProto,
) -> Tuple[Dict[str, List[List[int]]], Dict[str, numpy.dtype]]:
    """
    Creates a modified model to expose intermediate outputs and runs an ONNX Runtime
    InferenceSession to obtain the output shape of each node.

    :param model: an ONNX model
    :return: a list of NodeArg with their shape exposed
    """
    import onnxruntime

    model_copy = deepcopy(model)

    for node in model_copy.graph.node:
        intermediate_layer_value_info = make_empty_tensor_value_info(
            extract_node_id(node)
        )
        model_copy.graph.output.append(intermediate_layer_value_info)

    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = 3
    sess = onnxruntime.InferenceSession(model_copy.SerializeToString(), sess_options)

    input_value_dict = {}
    for input in model_copy.graph.input:
        input_shape = extract_shape(input)
        input_dtype = extract_dtype(input)

        input_shape = list(input_shape)
        input_shape[input_shape is None] = 1

        input_value_dict[input.name] = numpy.ones(input_shape, dtype=input_dtype)

    # Get shapes by running real values and saving outputs
    outputs = sess.run(None, input_value_dict)

    # Append inputs to list of nodes and outputs
    nodes = list(sess.get_outputs())
    for input in model_copy.graph.input:
        nodes.append(input)
        outputs.append(input_value_dict[input.name])

    output_shapes = {}
    output_dtypes = {}
    for node, output in zip(nodes, outputs):
        output_shapes[node.name] = (
            list(output.shape) if output is not None and len(output.shape) > 0 else None
        )
        output_dtypes[node.name] = output.dtype if output is not None else None

    return output_shapes, output_dtypes


def extract_nodes_shapes_and_dtypes_shape_inference(
    model: ModelProto,
) -> Tuple[Dict[str, List[Union[None, List[int]]]], Dict[str, numpy.dtype]]:
    """
    Creates a modified model to expose intermediate outputs and runs an ONNX shape
    inference to obtain the output shape of each node.

    NOTE: The ONNX docs on shape inference have the following
    disclaimer on shape inference:
    Shape inference is not guaranteed to be complete.
    In particular, some dynamic behaviors block the flow of shape inference,
    for example a Reshape to a dynamically-provide shape.
    Also, all operators are not required to have a shape inference implementation.

    :param model: an ONNX model
    :return: a list of NodeProto with their shape exposed
    """
    model_copy = deepcopy(model)

    for node in model_copy.graph.node:
        model_copy.graph.output.extend(
            [
                onnx.helper.make_tensor_value_info(
                    output, onnx.TensorProto.UNDEFINED, None
                )
                for output in node.output
            ]
        )

    if hasattr(onnx, "shape_inference"):
        model_copy = onnx.shape_inference.infer_shapes(model_copy)
    else:
        raise ModuleNotFoundError(
            "onnx.shape_inference not available for current version, "
            "please upgrade to use this functionality"
        )

    output_shapes = {}
    output_dtypes = {}
    for node in model_copy.graph.output:
        node_shape = extract_shape(node)
        dtype = extract_dtype(node)
        output_shapes[node.name] = (
            list(node_shape) if node_shape is not None and len(node_shape) > 0 else None
        )
        output_dtypes[node.name] = dtype

    return output_shapes, output_dtypes


def extract_nodes_shapes_and_dtypes(
    model: ModelProto,
) -> Tuple[Dict[str, List[List[int]]], Dict[str, numpy.dtype]]:
    """
    Uses ONNX Runtime or shape inference to infer output shapes and dtypes from model

    :param model: model to extract output values from
    :return: output shapes and output data types
    """
    output_shapes = None
    output_dtypes = None

    try:
        output_shapes, output_dtypes = extract_nodes_shapes_and_dtypes_ort(model)
    except Exception as err:
        _LOGGER.warning(f"Extracting shapes using ONNX Runtime session failed: {err}")

    if output_shapes is None or output_dtypes is None:
        _LOGGER.warning("Falling back to ONNX shape_inference")
        try:
            (
                output_shapes,
                output_dtypes,
            ) = extract_nodes_shapes_and_dtypes_shape_inference(model)
        except Exception as err:
            _LOGGER.warning(
                "Extracting shapes using ONNX shape_inference failed: {}".format(err)
            )

    return output_shapes, output_dtypes


def collate_output_shapes(
    model: ModelProto, output_shapes: Union[Dict[str, List[List[int]]], None]
) -> Dict[str, NodeShape]:
    """
    :param model: model whose shapes are being analyzed
    :param output_shapes: output shapes used to generate NodeShapes
    :return: a dictionary mapping node ids to NodeShapes
    """
    output_shapes = output_shapes if output_shapes is not None else {}

    # Maps NodeArg to its inputs
    node_to_inputs = {}
    for node in model.graph.node:
        node_to_inputs[extract_node_id(node)] = node.input

    input_shapes = {}
    for node_id in output_shapes.keys():
        if node_id not in node_to_inputs:
            continue
        input_shapes[node_id] = [
            output_shapes[input_node_id]
            for input_node_id in node_to_inputs[node_id]
            if input_node_id in output_shapes
            and output_shapes[input_node_id] is not None
        ]
        input_shapes[node_id] = (
            input_shapes[node_id] if len(input_shapes[node_id]) > 0 else None
        )

    node_shapes = {}
    for node_id in output_shapes.keys():
        node_shapes[node_id] = NodeShape(
            node_id,
            input_shapes[node_id] if node_id in input_shapes else None,
            [output_shapes[node_id]]
            if node_id in output_shapes and output_shapes[node_id] is not None
            else None,
        )

    def _fix_shapes(shapes: List[Union[List[Union[int, None, str]], None]]):
        if not shapes:
            return

        for shape in shapes:
            if not shape:
                continue

            for index, index_shape in enumerate(shape):
                try:
                    shape[index] = (
                        round(index_shape)
                        if isinstance(index_shape, float)
                        else int(index_shape)
                    )
                except Exception:
                    # not parsable as an int (none or string)
                    # set to None
                    shape[index] = None

    for node_id, node_shape in node_shapes.items():
        _fix_shapes(node_shape.input_shapes)
        _fix_shapes(node_shape.output_shapes)

    return node_shapes


def collate_output_dtypes(
    model: ModelProto, output_dtypes: Union[Dict[str, numpy.dtype], None]
) -> Dict[str, NodeDataType]:
    """
    :param model: model whose data types are being analyzed
    :param output_shapes: output data types used to generate NodeDataTypes
    :return: a dictionary mapping node ids to NodeDataTypes
    """
    output_dtypes = output_dtypes if output_dtypes is not None else {}

    # Maps NodeArg to its inputs
    node_to_inputs = {}
    for node in model.graph.node:
        node_to_inputs[extract_node_id(node)] = node.input

    input_dtypes = {}
    for node_id in output_dtypes.keys():
        if node_id not in node_to_inputs:
            continue
        input_dtypes[node_id] = [
            output_dtypes[input_node_id]
            for input_node_id in node_to_inputs[node_id]
            if input_node_id in output_dtypes
            and output_dtypes[input_node_id] is not None
        ]
        input_dtypes[node_id] = (
            input_dtypes[node_id] if len(input_dtypes[node_id]) > 0 else None
        )

    node_dtypes = {}
    for node_id in output_dtypes.keys():
        node_dtypes[node_id] = NodeDataType(
            node_id,
            input_dtypes[node_id] if node_id in input_dtypes else None,
            [output_dtypes[node_id]]
            if node_id in output_dtypes and output_dtypes[node_id] is not None
            else None,
        )

    return node_dtypes


def extract_node_shapes_and_dtypes(
    model: ModelProto,
) -> Tuple[Dict[str, NodeShape], Dict[str, NodeDataType]]:
    """
    Extracts the shape and dtype information for each node as NodeShape objects
    and numpy dtypes.

    :param model: the loaded onnx.ModelProto to extract node shape information from
    :return: a mapping of node id to a NodeShape object
    """

    # Obtains output shapes for each model's node
    output_shapes, output_dtypes = extract_nodes_shapes_and_dtypes(model)

    # Package output shapes into each node's inputs and outputs
    node_shapes = collate_output_shapes(model, output_shapes)
    node_dtypes = collate_output_dtypes(model, output_dtypes)

    return node_shapes, node_dtypes


def extract_dtype(proto: Any) -> numpy.dtype:
    """
    Extract data type info from a proto
    Used for reconstructing a node input for shape inference

    :param proto: the proto to get dtype info for
    :return: the numpy dtype of the tensor belonging to the proto
    """
    tensor_type = proto.type.tensor_type
    if not tensor_type.HasField("elem_type"):
        return None

    return TENSOR_TYPE_TO_NP_TYPE[proto.type.tensor_type.elem_type]


def extract_shape(proto: Any) -> Union[None, Tuple[Union[int, None], ...]]:
    """
    Extract the shape info from a proto.
    Convenient for inputs into a model for example to get the tensor dimension.

    :param proto: the proto to get tensor shape info for
    :return: a tuple containing shape info if found, else None
    """
    tensor_type = proto.type.tensor_type

    if not tensor_type.HasField("shape"):
        return None

    shape = []

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(None)

    return tuple(shape)
