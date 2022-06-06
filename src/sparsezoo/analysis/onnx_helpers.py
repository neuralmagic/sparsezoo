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
Utility / helper functions

NOTE: Adapted from sparseml/onnx/utils/helpers.py with minimal edits
"""

import logging
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import numpy
import onnx
from onnx import ModelProto, NodeProto
from onnx.helper import get_attribute_value, make_empty_tensor_value_info


_LOGGER = logging.getLogger(__name__)

__all__ = [
    "extract_node_id",
    "extract_node_shapes",
    "get_node_attributes",
    "NodeShape",
]


def extract_node_id(node: NodeProto) -> str:
    """
    Get the node id for a given node from an ONNX model.
    Grabs the first ouput id as the node id.
    This is because is guaranteed to be unique for this node by the ONNX spec.

    :param node: the node to grab an id for
    :return: the id for the node
    """
    outputs = node.output

    return str(outputs[0])


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


def extract_nodes_shapes_ort(model: ModelProto) -> Dict[str, List[List[int]]]:
    """
    Creates a modified model to expose intermediate outputs and runs an ONNX Runtime
    InferenceSession to obtain the output shape of each node.

    :param model: an ONNX model
    :return: a list of NodeArg with their shape exposed
    """
    import onnxruntime  # import protected by @require_onnxruntime()

    model_copy = deepcopy(model)

    for node in model_copy.graph.node:
        intermediate_layer_value_info = make_empty_tensor_value_info(
            extract_node_id(node)
        )
        model_copy.graph.output.append(intermediate_layer_value_info)

    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = 3
    sess = onnxruntime.InferenceSession(model_copy.SerializeToString(), sess_options)

    output_shapes = {}
    for node in sess.get_outputs() + sess.get_inputs():
        output_shapes[node.name] = (
            node.shape if node.shape is not None and len(node.shape) > 0 else None
        )
    return output_shapes


def extract_nodes_shapes_shape_inference(
    model: ModelProto,
) -> Dict[str, List[Union[None, List[int]]]]:
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
    for node in model_copy.graph.output:
        node_shape = extract_shape(node)
        output_shapes[node.name] = (
            list(node_shape) if node_shape is not None and len(node_shape) > 0 else None
        )

    return output_shapes


def extract_node_shapes(model: ModelProto) -> Dict[str, NodeShape]:
    """
    Extracts the shape information for each node as a NodeShape object.

    :param model: the loaded onnx.ModelProto to extract node shape information from
    :return: a mapping of node id to a NodeShape object
    """

    # Maps NodeArg to its inputs
    node_to_inputs = {}
    for node in model.graph.node:
        node_to_inputs[extract_node_id(node)] = node.input

    # Obtains output shapes for each model's node
    output_shapes = None

    try:
        output_shapes = extract_nodes_shapes_ort(model)
    except Exception as err:
        _LOGGER.warning(
            "Extracting shapes using ONNX Runtime session failed: {}".format(err)
        )

    if output_shapes is None:
        try:
            output_shapes = extract_nodes_shapes_shape_inference(model)
        except Exception as err:
            _LOGGER.warning(
                "Extracting shapes using ONNX shape_inference failed: {}".format(err)
            )

    # Obtains the input shapes for each node
    if output_shapes is None:
        output_shapes = {}

    input_shapes = {}

    for node in output_shapes.keys():
        if node not in node_to_inputs:
            continue
        input_shapes[node] = [
            output_shapes[input_node]
            for input_node in node_to_inputs[node]
            if input_node in output_shapes and output_shapes[input_node] is not None
        ]
        input_shapes[node] = input_shapes[node] if len(input_shapes[node]) > 0 else None

    # Combines shape information into mapping of node id to a NodeShape object
    node_shapes = {}
    for node in output_shapes.keys():
        node_shapes[node] = NodeShape(
            node,
            input_shapes[node] if node in input_shapes else None,
            [output_shapes[node]]
            if node in output_shapes and output_shapes[node] is not None
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


"""
Simple named tuple for mapping a node value to the init name it came from
"""
NodeParam = NamedTuple("NodeParam", [("name", str), ("val", numpy.ndarray)])

def get_node_attributes(node: NodeProto) -> Dict[str, Any]:
    """
    :param node: the ONNX node to get the attibutes for
    :return: a dictionary containing all attributes for the node
    """
    attributes = reduce(
        lambda accum, attribute: accum.update(
            {attribute.name: get_attribute_value(attribute)}
        )
        or accum,
        node.attribute,
        {},
    )

    for key in list(attributes.keys()):
        val = attributes[key]

        if not (
            isinstance(val, int)
            or isinstance(val, float)
            or isinstance(val, str)
            or isinstance(val, list)
            or isinstance(val, dict)
        ):
            attributes[key] = None

    return attributes


''' TODO: Remove
def get_kernel_shape(attributes: Dict[str, Any]) -> Union[List[float], None]:
    """
    Get the kernel shape from a dictionary of a model's attributes

    :param attributes: a dictionary of a model's attributes
    :return: the kernel shape if attribute contains either the kernel or
        kernel_shape field, otherwise None
    """
    if "kernel" in attributes:
        return attributes["kernel"]
    elif "kernel_shape" in attributes:
        return attributes["kernel_shape"]
    else:
        return None


def calculate_num_operations(
    op_type: str,
    input_shape: Union[List[List], None] = None,
    output_shape: Union[List[List], None] = None,
    weight_shape: Union[List, None] = None,
    kernel_shape: Union[List, None] = None,
    bias_shape: Union[List, None] = None,
    attributes: Union[None, Dict[str, Any]] = None,
) -> Union[float, None]:
    """
    Calculate flops based on operation type and shape of certain attributes.
    If any fields necessary in operation are set to None, will return None

    :param op_type: Operation type of flop calculation
    :param input_shape: List of input shapes of operation
    :param output_shape: List of output shapes of operation
    :param weight_shape: Shape of weights in operation if any, else None
    :param kernel_shape: Shape of kernel in operation if any, else None
    :param bias_shape: Shape of bias in operation if any, else None
    :param attributes: The node attributes if any, else None
    :return: The amount of floating point operations in the operation
    """
    input_shape = _array_as_numeric(input_shape)
    output_shape = _array_as_numeric(output_shape)
    weight_shape = _array_as_numeric(weight_shape)
    kernel_shape = _array_as_numeric(kernel_shape)
    bias_shape = _array_as_numeric(bias_shape)

    if (
        op_type == "Add"
        or op_type == "Mul"
        or op_type == "Div"
        or op_type == "Sub"
        or op_type == "Clip"
    ):
        num_operations = _numpy_prod_with_none_check(output_shape)
    elif (
        op_type == "Relu"
        or op_type == "LeakyRelu"
        or op_type == "Sigmoid"
        or op_type == "Tanh"
        or op_type == "BatchNormalization"
    ):
        num_operations = _numpy_prod_with_none_check(output_shape)
    elif op_type == "GlobalAveragePool" or op_type == "GlobalMaxPool":
        num_operations = _numpy_prod_with_none_check(input_shape)
    elif op_type == "MaxPool" or op_type == "AveragePool":
        num_operations = (
            numpy.prod(output_shape) * numpy.prod(kernel_shape)
            if output_shape is not None and kernel_shape is not None
            else None
        )
    elif op_type in ["MatMul", "MatMulInteger", "QLinearMatMul"]:
        num_operations = _calculate_num_ops_matmul(
            op_type,
            input_shape=input_shape,
            output_shape=output_shape,
            weight_shape=weight_shape,
        )
    elif op_type == "Gemm":
        num_operations = _numpy_prod_with_none_check(weight_shape)
        num_operations = num_operations * 2 if num_operations is not None else None
    elif op_type in ["Conv", "ConvInteger", "QLinearConv"]:
        # num values in kernel * num output maps * output spatial dimensions
        # But I think it should actually be times 2 b/c op + equals
        num_operations = (
            numpy.prod(kernel_shape) * weight_shape[1] * numpy.prod(output_shape) * 2
            if kernel_shape is not None
            and weight_shape is not None
            and output_shape is not None
            else None
        )

        if (
            num_operations
            and attributes
            and "group" in attributes
            and attributes["group"]
            and attributes["group"] > 1
        ):
            # adjust flops for group / depthwise convolutions
            num_operations = num_operations / attributes["group"]
    else:
        num_operations = None

    if num_operations is not None and bias_shape is not None:
        if op_type in ["Conv", "ConvInteger", "QLinearConv"]:
            num_operations += (
                numpy.prod(bias_shape) * output_shape[0][-1] * output_shape[0][-2] * 2
            )
        else:
            num_operations += numpy.prod(bias_shape) * 2

    return num_operations


def _calculate_num_ops_matmul(
    op_type: str,
    input_shape: Union[List[List], None] = None,
    output_shape: Union[List[List], None] = None,
    weight_shape: Union[List, None] = None,
) -> Union[float, None]:
    """
    Calculates flops in an ONNX MatMul operation.

    If input shape only contains 1 input, in otherwords the value of the
    first index is 1, then the matrix operation is treated as a Gemm operation.

    Otherwise the operation is treated like a NumPy operation.

    Will return none if any required value is set to None

    :param op_type: Operation type of flop calculation
    :param input_shape: List of input shapes of operation
    :param output_shape: List of output shapes of operation
    :param weight_shape: Shape of weights in operation if any, else None
    :return: The amount of floating point operations in the operation
    """
    flops = None
    if (
        input_shape is not None
        and output_shape is not None
        and len(input_shape) > 1
        and input_shape[0][-1] == input_shape[1][-2]
    ):
        matrix_ops = (
            input_shape[0][-2] * input_shape[1][-1] * (2 * input_shape[0][-1] - 1)
        )
        flops = numpy.prod(output_shape[0][:-2]) * matrix_ops
    elif input_shape is not None and len(input_shape) == 1:
        flops = _numpy_prod_with_none_check(weight_shape)
        flops = flops * 2 if flops is not None else None
    return flops


def _numpy_prod_with_none_check(array: Union[List, None]) -> Union[float, None]:
    """
    :param array: an array like list
    :return: the product of the array if array is not None otherwise return None
    """
    return numpy.prod(array) if array is not None else None


def _attempt_cast_as_float(value: Any) -> float:
    """
    :param vale: a value
    :return: the value as a float if casting is possible, otherwise return 1
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 1.0


def _array_as_numeric(array: Union[List, None]) -> Union[List, None]:
    """
    :param array: an array like list
    :return: the array with any non numeric or None values replaced with 1
        if array itself is not None, otherwise return None
    """
    if array is None:
        return None

    array = numpy.array(array, dtype=object)
    # Check if the array datatype is a number
    if numpy.issubdtype(array.dtype, numpy.number):
        return array
    else:
        to_float = numpy.vectorize(_attempt_cast_as_float)
        return to_float(array)

'''
