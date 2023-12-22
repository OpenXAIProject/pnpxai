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

@dataclass
class NodeInfo:
    """
    NodeInfo is a dataclass representing a node: `torch.fx.Node`
    in a graph: `torch.fx.Graph`. There are 3 types of constructor:
    `from_node`, `from_module`, and `from_function`.

    - `from_node` clones a node in a graph
    - `from_module` generates an instance to replace with a node
       or to be inserted in a graph from a module
    - `from_function` generates an instance to replace with a node
       or to be inserted in a graph from a function
    
    Attributions
    - `opcode` is an operation code of a node, which is one of
      [
        "placeholder", "get_attr", "call_function",
        "call_module", "call_method", "output",
      ]
    - `name` is a name of node auto-assigned by `torch.fx.symbolic_graph`
    - `target` is an accessible name of node
    """
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
        self._mode = "from_node"
        self._set_node(n.graph)
        return self

    @classmethod
    def from_module(cls, module: torch.nn.Module, **kwargs):
        self = cls(
            opcode = "call_module",
            name = None,
            target = None,
        )
        self._mode = "from_module"
        self._set_operator(module)
        self._kwargs = kwargs
        return self

    @classmethod
    def from_function(cls, func: Callable, **kwargs):
        self = cls(
            opcode = "call_function",
            name = None,
            target = None,
        )
        self._mode = "from_function"
        self._set_operator(func)
        self._kwargs = kwargs
        return self

    # set original node from the graph
    def _set_node(self, graph):
        assert self._mode == "from_node"
        # [GH] no `get_node` method in `torch.fx.Graph`. just find.
        for n in graph.nodes:
            if n.name == self.name:
                self._node = n
    
    def _set_operator(self, operator: Callable):
        assert self._mode != "from_node"
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
            for k, v in self._node.kwargs.items()
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
                root_module = SUPPORTED_FUNCTION_MODULES.get(targets.pop(0))
                if root_module:
                    return self._get_operator(targets, root_module)
            return
        return self._operator
    
    @property
    def owning_module(self) -> Optional[str]:
        if self.opcode in ["call_module", "call_function"]:
            if self.meta.get("nn_module_stack"):
                nm = next(iter(self.meta["nn_module_stack"]))
                return nm
        return
        
    # convert data format
    def to_dict(self):
        return {**asdict(self), "operator": self.operator}
    
    # [TODO] to_json for visualization
    def to_json_serializable(self):
        pass

class ModelArchitecture:
    """
    ModelArchitecture is a helper class to manipulate a graph generated
    by `torch.fx.symbolic_graph`.
    """
    def __init__(self, model):
        self.model = model
        self.traced_model = symbolic_trace(model)

        self._replacing = False
    
    def list_nodes(self) -> List[NodeInfo]:
        """
        List all nodes in graph.
        """
        return [NodeInfo.from_node(n) for n in self.traced_model.graph.nodes]
    
    def get_node(self, name: str) -> NodeInfo:
        """
        Get a node in graph by name.
        """
        for n in self.traced_model.graph.nodes:
            if n.name == name:
                return NodeInfo.from_node(n)
        return # [TODO] Error for no result?
        
    def find_node(self, filter_func: Callable, root: Optional[NodeInfo]=None, all=False):
        """
        Find a node satisfying `filter_func` in graph.
        Searching is started from `root` node.
        """
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
    
    def _validate_new_node(self, new_node: NodeInfo):
        if new_node.name is not None:
            exists = self.get_node(name=new_node.name)
            assert not exists, f"A node named {new_node.name} already exists."
        return True

    def _ensure_graph(self) -> None:
        self.traced_model.graph.lint()
        self.traced_model.recompile()
    
    def replace_node(self, node: NodeInfo, new_node: NodeInfo) -> NodeInfo:
        """
        Replace a `node` with `new_node`.
        """
        self._replacing = True
        self._validate_new_node(new_node)
        try:
            if new_node._mode == "from_module":
                new_node.name = f"{REPLACE_PREFIX}{node.name}"
            inserted = self.insert_node(new_node, base_node=node)
            self.traced_model.graph.erase_node(node._node)
            self._ensure_graph()
        finally:
            self._replacing = False
        return inserted
    
    def insert_node(self, new_node: NodeInfo, base_node: NodeInfo, before=False) -> NodeInfo:
        """
        Insert a `new_node` after `base_node`.
        """
        self._validate_new_node(new_node)
        inserting = self.traced_model.graph.inserting_before if before else self.traced_model.graph.inserting_after
        if self._replacing:
            _inserted_args = base_node._node.args
            _inserted_kwargs = base_node._node.kwargs
            pass
        elif before:
            _inserted_args = tuple(arg for arg in base_node._node.args if isinstance(arg, Node))
            _inserted_kwargs = {kw: arg for kw, arg in base_node._node.kwargs.items() if isinstance(arg, Node)}
            _inserted_kwargs = {**_inserted_kwargs, **new_node._kwargs}
        else:
            _inserted_args = (base_node,)
            _inserted_kwargs = new_node._kwargs

        # [TODO] validation for new_node: if new_node._mode=="from_module" , new_node.name exists
        # insert
        with inserting(base_node._node):
            if new_node._mode == "from_module":
                self.traced_model.add_submodule(new_node.name, new_node.operator)
                _inserted = self.traced_model.graph.call_module(
                    new_node.name,
                    _inserted_args,
                    _inserted_kwargs,
                )
            elif new_node._mode == "from_function":
                _inserted = self.traced_model.graph.call_function(
                    new_node.operator,
                    _inserted_args,
                    _inserted_kwargs,
                )
        
        if self._replacing or not before:
            base_node._node.replace_all_uses_with(_inserted)
            pass
        elif before:
            _not_inserted_args = [arg for arg in base_node._node.args if not isinstance(arg, Node)]
            base_node._node.args = tuple([_inserted] + _not_inserted_args)
            base_node._node.kwargs = {kw: arg for kw, arg in base_node._node.kwargs.items() if not isinstance(arg, Node)}
        self._ensure_graph()
        return NodeInfo.from_node(_inserted)        

    def to_dict(self):
        """
        Convert the model architecture into dict in order to visualize.
        """
        nodes = []
        edges = []
        for n in self.list_nodes():
            nodes.append(n.to_dict())
            edges += [{"source": n.name, "target": u.name} for u in n.users]
        return {"nodes": nodes, "edges": edges}
