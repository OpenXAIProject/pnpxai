import _operator
from dataclasses import dataclass, asdict
from typing import Literal, List, Tuple, Optional, Callable, Union

import torch
from torch.fx import Node, Graph, symbolic_trace

from .filters import conv_filter, pool_filter

SUPPORTED_FUNCTION_MODULES = {
    "torch": torch,
    "operator": _operator,
}


@dataclass
class NodeInfo:
    opcode: Literal[
        "placeholder", "get_attr", "call_function",
        "call_module", "call_method", "output",
    ]
    name: str
    target: Union[Callable, str]

    @classmethod
    def from_node(cls, n: Node):
        self = cls(
            opcode = n.op,
            name = n.name,
            target = n._pretty_print_target(n.target),
        )
        self._set_node(n.graph)
        return self
    
    # set original node from the graph
    def _set_node(self, graph):
        # [GH] no `get_node` method in `torch.fx.Graph`. just find.
        for n in graph.nodes:
            if n.name == self.name:
                self._node = n
    
    # cloning main attributions
    @property
    def meta(self):
        return self._node.meta
    
    @property
    def args(self):
        return tuple([
            NodeInfo.from_node(a) if isinstance(a, Node) else a
            for a in self._node.args
        ])
    
    @property
    def kwargs(self):
        return {
            k: NodeInfo.from_node(n) if isinstance(v, Node) else v
            for k, v in self._node.kwargs
        }
    
    @property
    def users(self):
        return tuple([
            NodeInfo.from_node(u)
            for u in self._node.users.keys()
        ])
    
    @property
    def next(self):
        if self._node.next.op == "root":
            return
        return NodeInfo.from_node(self._node.next)
    
    @property
    def prev(self):
        if self._node.prev.op == "root":
            return
        return NodeInfo.from_node(self._node.prev)
    
    # additional properties for detection
    @property
    def operator(self) -> Optional[Callable]:
        if self.opcode == "call_module":
            return self._get_operator(self.target.split("."))
        elif self.opcode == "call_function":
            targets = self.target.split(".")
            root_module = SUPPORTED_FUNCTION_MODULES[targets.pop(0)]
            return self._get_operator(targets, root_module)
        return
    
    @property
    def owning_module(self) -> Optional[Tuple[str, torch.nn.Module]]:
        if self.opcode in ["call_module", "call_function"]:
            if self.meta.get("nn_module_stack"):
                nm = next(reversed(self.meta["nn_module_stack"]))
                return nm, self._get_operator(nm.split("."))
        return
    
    def _get_operator(self, targets: List[str], root_module=None):
        operator = root_module if root_module else self._node.graph.owning_module
        for s in targets:
            operator = getattr(operator, s)
        return operator
    
    # convert data format
    def to_dict(self):
        return asdict(self)
    
    # [TODO] to_json for visualization
    def to_json_serializable(self):
        pass

class ModelArchitecture:
    def __init__(self, graph: Graph):
        self.graph = graph
    
    @classmethod
    def from_model(cls, model: torch.nn.Module):
        traced_model = symbolic_trace(model)
        return cls(graph=traced_model.graph)
    
    def list_nodes(self) -> List[NodeInfo]:
        return [NodeInfo.from_node(n) for n in self.graph.nodes]
    
    def get_node(self, name: str) -> NodeInfo:
        for n in self.graph.nodes:
            if n.name == name:
                return NodeInfo.from_node(n)
        return # [TODO] Error for no result?
        
    def find_node(self, filter_func: Callable, root: Optional[NodeInfo]=None, all=False):
        # [TODO] breadth first / depth first
        if not root:
            root = self.list_nodes()[0]
        def _find_node(n):
            if n == None:
                return None
            if filter_func(n):
                return n
            else:
                return _find_node(n.next)
        if not all:
            return _find_node(root)
        nodes = []
        n = root
        while True:
            found = _find_node(n)
            if not found:
                break
            nodes.append(found)
            n = found.next
        return nodes
    
    def find_cam_target_module(self) -> Optional[Tuple[str, torch.nn.Module]]:
        # filters to find target node
        first_conv_node = self.find_node(conv_filter)
        if not first_conv_node:
            return
        pool_nodes = self.find_node(pool_filter, all=True)
        if not pool_nodes:
            return
        return pool_nodes[-1].prev.owning_module
    
    def to_dict(self):
        nodes = []
        edges = []
        for n in self.list_nodes():
            nodes.append(n.to_dict())
            edges += [{"source": n.name, "target": u.name} for u in n.users]
        return {"nodes": nodes, "edges": edges}
