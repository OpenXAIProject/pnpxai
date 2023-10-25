from dataclasses import dataclass, asdict
from typing import Literal, List, Dict, Tuple, Optional, Callable, Union

from torch import nn
from torch.fx import Node, Graph, symbolic_trace


@dataclass
class NodeInfo:
    opcode: Literal[
        "placeholder", "get_attr", "call_function",
        "call_module", "call_method", "output",
    ]
    name: str
    target: Union[Callable, str]
    # args: Tuple
    # kwargs: Dict

    @classmethod
    def from_node(cls, n: Node):
        self = cls(
            opcode = n.op,
            name = n.name,
            target = n.target,
        )
        self._set_node(n.graph)
        return self
    
    # set original node from the graph
    def _set_node(self, graph) -> Node:
        # [GH] no `get_node` method in `torch.fx.Graph`. just find.
        for n in graph.nodes:
            if n.name == self.name:
                self._node = n
    
    # cloning main attributions
    @property
    def args(self):
        return tuple([
            NodeInfo.from_node(n) if isinstance(a, Node) else a
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
    def operator(self, base_model: Optional[nn.Module]=None) -> Optional[Callable]:
        if self.opcode == "call_module":
            target = model if base_model else self._node.graph.owning_module
            for s in self.target.split("."):
                target = getattr(target, s)
            return target
        elif self.opcode == "call_function":
            return self.target # target is Callable itself
        # self.op == "call_method": Usually, call a method of torch.Tensor but whatif not
        # self.op in ["input", "output", "get_attr"]: non-operator output
        return
    
    # convert data format
    def to_dict(self):
        return asdict(self)
    
    # [TODO] to_json for visualization
    def to_json(self):
        pass


class ModelArchitecture:
    def __init__(self, graph: Graph):
        self.graph = graph
    
    @classmethod
    def from_model(cls, model: nn.Module):
        traced_model = symbolic_trace(model)
        return cls(graph=traced_model.graph)
    
    def list_nodes(self) -> List[NodeInfo]:
        return [NodeInfo.from_node(n) for n in self.graph.nodes]
    
    def get_node(self, name: str) -> NodeInfo:
        for n in self.graph.nodes:
            if n.name == name:
                return NodeInfo.from_node(n)
        return # [TODO] Error for no result?
        
    def find_node(self, filter_func: Callable, base_node: Optional[NodeInfo]=None, all=False):
        if not base_node:
            base_node = self.list_nodes()[0]
        def _find_node(n):
            if n == None:
                return None
            if filter_func(n):
                return n
            else:
                return _find_node(n.next)
        if not all:
            return _find_node(base_node)
        nodes = []
        n = base_node
        while True:
            found = _find_node(n)
            if not found:
                break
            nodes.append(found)
            n = found.next
        return nodes
    
    def find_cam_target_node(self):
        # filters to find target node
        conv_filter = lambda n: (
            n.opcode == "call_module"
            and n.operator.__module__ == "torch.nn.modules.conv"
        )
        pool_filter = lambda n: (
            n.opcode == "call_module"
            and n.operator.__module__ == "torch.nn.modules.pooling"
            and len(n.users) == 1
        )
        first_conv_node = self.find_node(conv_filter)
        if not first_conv_node:
            return
        final_pool_node = self.find_node(pool_filter)
        if not final_pool_node:
            return
        return final_pool_node.prev
