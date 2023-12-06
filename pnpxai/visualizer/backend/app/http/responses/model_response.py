from typing import Tuple, List, Dict

from pnpxai.explainers.utils.operation_graph import OperationGraph, OperationNode
from pnpxai.visualizer.backend.app.core.constants import APIItems
from pnpxai.visualizer.backend.app.core.generics import Response


class ModelSchema(Response):
    @classmethod
    def generate_nodes_edges_list(cls, model) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        graph = OperationGraph(model)
        nodes = {}
        edges = {}
        stack = [graph.root]
        while len(stack) > 0:
            node: OperationNode = stack.pop()
            nodes[node.name] = {
                APIItems.ID.value: node.name
            }

            for next_node in node.next_nodes:
                stack.append(next_node)
                if f"{node.name}{next_node.name}" in edges or f"{next_node.name}{node.name}" in edges:
                    continue

                key = f"{node.name}{next_node.name}"
                edges[key] = {
                    APIItems.ID.value: key,
                    APIItems.SOURCE.value: node.name,
                    APIItems.TARGET.value: next_node.name,
                }
        node_list = list(nodes.values())
        edge_list = list(edges.values())

        return node_list, edge_list

    @classmethod
    def to_dict(cls, model):
        nodes, edges = cls.generate_nodes_edges_list(model)
        return {
            APIItems.NODES.value: nodes,
            APIItems.EDGES.value: edges
        }
