import _operator
from dataclasses import dataclass, asdict
from typing import Literal, List, Tuple, Optional, Callable, Union

import torch
from torch.fx import Node, symbolic_trace

SUPPORTED_FUNCTION_MODULES = {
    "torch": torch,
    "operator": _operator,
}

REPLACE_PREFIX = "_replaced_"

@dataclass #(init=True)
class NodeInfo:
    opcode: Literal[
        "placeholder", "get_attr", "call_function",
        "call_module", "call_method", "output",
    ]
    name: str
    target: Union[Callable, str]

    # def __init__(self, opcode, name, target, _operator=None, _from_node=False):
    #     self.opcode = opcode
    #     self.name = name
    #     self.target = target

    #     self._from_node = _from_node
    #     self._operator = _operator

    @classmethod
    def from_node(cls, n: Node):
        self = cls(
            opcode = n.op,
            name = n.name,
            target = n._pretty_print_target(n.target),
        )
        self._mode = "from_node"
        self._set_node(n.graph)
        return self

    @classmethod
    def from_module(cls, m: torch.nn.Module):
        self = cls(
            opcode = "call_module",
            name = None,
            target = None,
        )
        self._mode = "from_module"
        self._set_operator(m)
        return self

    # set original node from the graph
    def _set_node(self, graph):
        assert self._mode == "from_node"
        # [GH] no `get_node` method in `torch.fx.Graph`. just find.
        for n in graph.nodes:
            if n.name == self.name:
                self._node = n
    
    def _set_operator(self, operator: Callable):
        assert self._mode == "from_module"
        self._valid_operator(operator)
        self._operator = operator
        return self

    # [TODO] validation for operator
    def _valid_operator(self, operator: Callable):
        pass
    
    def _get_operator(self, targets: List[str], root_module=None):
        operator = root_module if root_module else self._node.graph.owning_module
        for s in targets:
            operator = getattr(operator, s)
        return operator
    
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
            k: NodeInfo.from_node(v) if isinstance(v, Node) else v
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
        if self._mode == "from_node":
            if self.opcode == "call_module":
                return self._get_operator(self.target.split("."))
            elif self.opcode == "call_function":
                targets = self.target.split(".")
                root_module = SUPPORTED_FUNCTION_MODULES[targets.pop(0)]
                return self._get_operator(targets, root_module)
            return
        return self._operator
    
    @property
    def owning_module(self) -> Optional[str]:
        if self.opcode in ["call_module", "call_function"]:
            if self.meta.get("nn_module_stack"):
                nm = next(reversed(self.meta["nn_module_stack"]))
                return nm
        return
        
    # convert data format
    def to_dict(self):
        return asdict(self)
    
    # [TODO] to_json for visualization
    def to_json_serializable(self):
        pass

class ModelArchitecture:
    def __init__(self, model):
        self.model = model
        self.traced_model = symbolic_trace(model)
    
    def list_nodes(self) -> List[NodeInfo]:
        return [NodeInfo.from_node(n) for n in self.traced_model.graph.nodes]
    
    def get_node(self, name: str) -> NodeInfo:
        for n in self.traced_model.graph.nodes:
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
    
    def replace_node(self, node: NodeInfo, new_node: NodeInfo):
        # validations
        assert new_node.opcode == "call_module", "call_module only"
        assert new_node._operator, "Must set operator for new_node: new_node.set_operator(operator)."
        if new_node.name is not None:
            exists = self.get_node(name=new_node.name)
            assert exists, f"A node named {new_node.name} already exists."
        
        # replace
        with self.traced_model.graph.inserting_after(node._node):
            new_name = new_node.name if new_node.name else f"{REPLACE_PREFIX}{node.name}"
            self.traced_model.add_submodule(new_name, new_node.operator)
            _new_node = self.traced_model.graph.call_module(
                new_name,
                node._node.args,
                node._node.kwargs,
            )
            node._node.replace_all_uses_with(_new_node)
        self.traced_model.graph.erase_node(node._node)
        self.traced_model.graph.lint()
        self.traced_model.recompile()
        return self
            
    def to_dict(self):
        nodes = []
        edges = []
        for n in self.list_nodes():
            nodes.append(n.to_dict())
            edges += [{"source": n.name, "target": u.name} for u in n.users]
        return {"nodes": nodes, "edges": edges}
