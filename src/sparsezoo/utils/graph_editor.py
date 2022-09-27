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
Helper functions to edit ONNX Graphs.
"""

from collections import defaultdict
from typing import Iterable, List, Optional, Union

from onnx import ModelProto, NodeProto, TensorProto


__all__ = [
    "ONNXGraph",
]


class ONNXGraph(object):
    """
    Class for quick look-up of ONNX graph nodes and initializers. If graph state
    changes outside of ONNXGraph class functions, update() should be called.

    :param model: the ONNX graph to represent
    """

    def __init__(self, model: ModelProto):
        self._model = model
        self._output_id_to_node = {}
        self._input_id_to_nodes = defaultdict(list)
        self._name_to_initializer = {}

        self.update()

    @property
    def model(self) -> ModelProto:
        return self._model

    @property
    def nodes(self) -> Iterable[NodeProto]:
        """
        :return: ordered collection of nodes in this graph
        """
        return self._model.graph.node

    def update(self, model: Optional[ModelProto] = None):
        """
        Update the graph state based on the model this graph represents or
        the given model.

        :param model: model to represent. defaults to current loaded model state
        """
        self._model = model or self._model

        # nodes
        self._output_id_to_node = {}
        self._input_id_to_nodes = defaultdict(list)
        for node in self._model.graph.node:
            self._store_node_edges(node)

        # initializers
        self._name_to_initializer = {
            init.name: init for init in self._model.graph.initializer
        }

    def get_init_by_name(
        self,
        name: str,
        allow_optional: bool = True,
    ) -> Optional[TensorProto]:
        """
        :param name: name of initializer
        :param allow_optional: if True and the given name is not found as an
            initializer, None will be returned. Otherwise a KeyError will be raised
        :return: tensor of initializer with given name, returns None if the name does
            not exist in the cached graph
        """
        init = self._name_to_initializer.get(name, None)
        if not allow_optional and init is None:
            raise KeyError(f"Unable to find initializer {name} in ONNX model")
        return init

    def get_init_names(self):  # TODO
        """
        TODO
        """
        return self._name_to_initializer.keys()

    def get_node_by_output_id(self, id: str) -> Optional[TensorProto]:
        """
        :param id: name of output id of node
        :return: the associated node if it is present in the graph, None otherwise
        """
        return self._output_id_to_node.get(id)

    def get_node_parents(
        self, node: NodeProto
    ) -> List[Union[NodeProto, TensorProto, None]]:
        """
        :param node: node to get the input objects for
        :return: input nodes or tensors of this node in order. if an input does not
            exist, None will be returned in its place
        """
        inputs = []
        for input_id in node.input:
            inp = None
            if input_id in self._output_id_to_node:
                inp = self._output_id_to_node[input_id]
            elif input_id in self._name_to_initializer:
                inp = self._name_to_initializer[input_id]
            inputs.append(inp)
        return inputs

    def get_node_single_parent(
        self, node: NodeProto, index: int
    ) -> Union[NodeProto, None]:
        """
        :param node: the node to get the parent node of
        :param index: choose which input to search
        :return: parent of node if it only has one parent, otherwise None
        """
        input_id = node.input[index]
        if input_id not in self._output_id_to_node:
            return None
        return self._output_id_to_node[input_id]

    def get_node_children(self, node: NodeProto) -> List[NodeProto]:
        """
        :param node: the node to get the children node of
        :return: list of nodes that include this node as an output
        """
        children = []
        for output_id in node.output:
            children.extend(self._input_id_to_nodes[output_id])
        return children

    def get_node_single_child(self, node: NodeProto) -> Union[NodeProto, None]:
        """
        :param node: the node to get the child node of
        :return: child of node if it only has one child, otherwise None
        """
        children = self.get_node_children(node)
        return children[0] if len(children) == 1 else None

    def add_node(self, node: NodeProto):
        """
        Adds the given node to the model and graph state

        :param node: node to add to the model
        """
        self._model.graph.node.append(node)
        self._store_node_edges(node)

    def update_node_input(
        self, node: NodeProto, input_id: str, input_idx: Optional[int] = None
    ):
        """
        :param node: node to update the inputs of
        :param input_id: new input_id to attach to the node
        :param input_idx: optional index of the node input list to update,
            if none is given, the new input id will be appended to the input list
        """
        if input_idx is not None:
            if node in self._input_id_to_nodes[node.input[input_idx]]:
                self._input_id_to_nodes[node.input[input_idx]].remove(node)
            node.input[input_idx] = input_id
        else:
            node.input.append(input_id)
        self._input_id_to_nodes[input_id].append(node)

    def delete_node(self, node: NodeProto):
        """
        deletes the given node from the graph

        :param node: node to delete
        """
        self._model.graph.node.remove(node)
        self._delete_node_edges(node)

    def delete_nodes(self, nodes: List[NodeProto]):
        """
        deletes the given nodes from the graph
        :param nodes: list of nodes to delete
        """
        node_ouptut_ids_to_delete = {node.output[0] for node in nodes}
        nodes_to_keep = []
        for node in self._model.graph.node:
            if node.output[0] in node_ouptut_ids_to_delete:
                self._delete_node_edges(node)
            else:
                nodes_to_keep.append(node)
        self._model.graph.ClearField("node")
        self._model.graph.node.extend(nodes_to_keep)

    def delete_initializers(self, initializers: List[Union[str, TensorProto]]):
        """
        deletes the given initializers from the model

        :param initializers: list of initializers or initializer names to delete
        """
        inits_to_delete = {
            init if isinstance(init, str) else init.name for init in initializers
        }
        inits_to_keep = []
        for init in self._model.graph.initializer:
            if init.name in inits_to_delete:
                # keep edge reference if nodes in the graph still point to the
                # initializer name
                if not self._input_id_to_nodes[init.name]:
                    del self._input_id_to_nodes[init.name]
                del self._name_to_initializer[init.name]
            else:
                inits_to_keep.append(init)
        self._model.graph.ClearField("initializer")
        self._model.graph.initializer.extend(inits_to_keep)

    def delete_unused_initializers(self):
        """
        deletes tensors in the initializer list that are not listed as inputs to any
        node in the current graph state or directly passed as model outputs
        """
        output_names = {out.name for out in self._model.graph.output}
        self.delete_initializers(
            [
                init
                for init in self._model.graph.initializer
                if not self._input_id_to_nodes[init.name]
                and (init.name not in output_names)
            ]
        )  # delete inits that have no edge

    def _store_node_edges(self, node: NodeProto):
        for output_id in node.output:
            self._output_id_to_node[output_id] = node
        for input_id in node.input:
            self._input_id_to_nodes[input_id].append(node)

    def _delete_node_edges(self, node: NodeProto):
        # remove node edges from cache
        for output_id in node.output:
            del self._output_id_to_node[output_id]
        for input_id in node.input:
            self._input_id_to_nodes[input_id].remove(node)
