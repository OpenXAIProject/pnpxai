import _operator
from dataclasses import dataclass, asdict
from typing import Literal, List, Optional, Callable, Union

import torch
from torch.fx import Node, symbolic_trace, Tracer, GraphModule

SUPPORTED_FUNCTION_MODULES = {
    "torch": torch,
    "operator": _operator,
}

REPLACE_PREFIX = "_replaced_"


@dataclass
class NodeInfo:
    """
    Represents information about a node in a computation graph.

    Attributes:
    - opcode (Literal[str]): The operation code associated with the node.
    - name (str): The name of the node.
    - target (Union[Callable, str]): The target of the node, which could be a callable object or a string.
    """
    opcode: Literal[
        "placeholder", "get_attr", "call_function",
        "call_module", "call_method", "output",
    ]
    name: str
    target: Union[Callable, str]

    @classmethod
    def from_node(cls, n: Node):
        """
        Constructs a NodeInfo object from a given Node.

        Args:
        - n (Node): The node from which to construct the NodeInfo object.

        Returns:
        - NodeInfo: The NodeInfo object constructed from the given Node.
        """
        self = cls(
            opcode=n.op,
            name=n.name,
            target=n._pretty_print_target(n.target),
        )
        self._mode = "from_node"
        self._set_node(n.graph)
        return self

    @classmethod
    def from_module(cls, module: torch.nn.Module, **kwargs):
        """
        Constructs a NodeInfo object from a given torch.nn.Module.

        Args:
        - module (torch.nn.Module): The module from which to construct the NodeInfo object.
        - **kwargs: Additional keyword arguments.

        Returns:
        - NodeInfo: The NodeInfo object constructed from the given module.
        """
        self = cls(
            opcode="call_module",
            name=None,
            target=None,
        )
        self._mode = "from_module"
        self._set_operator(module)
        self._kwargs = kwargs
        return self

    @classmethod
    def from_function(cls, func: Callable, **kwargs):
        """
        Constructs a NodeInfo object from a given callable function.

        Args:
        - func (Callable): The function from which to construct the NodeInfo object.
        - **kwargs: Additional keyword arguments.

        Returns:
        - NodeInfo: The NodeInfo object constructed from the given function.
        """
        self = cls(
            opcode="call_function",
            name=None,
            target=None,
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
        """
        Property to access the meta information of the node.

        Returns:
        - Any: The meta information of the node.
        """
        return self._node.meta

    @property
    def args(self):
        """
        Property to access the arguments of the node.

        Returns:
        - tuple: The arguments of the node.
        """
        return tuple([
            NodeInfo.from_node(a) if isinstance(a, Node) else a
            for a in self._node.args
        ])

    @property
    def kwargs(self):
        """
        Property to access the keyword arguments of the node.

        Returns:
        - dict: The keyword arguments of the node.
        """
        return {
            k: NodeInfo.from_node(v) if isinstance(v, Node) else v
            for k, v in self._node.kwargs.items()
        }

    @property
    def users(self):
        """
        Property to access the users of the node.

        Returns:
        - tuple: The users of the node.
        """
        return tuple([
            NodeInfo.from_node(u)
            for u in self._node.users.keys()
        ])

    @property
    def next(self):
        """
        Property to access the next node.

        Returns:
        - NodeInfo: The next node.
        """
        if self._node.next.op == "root":
            return
        return NodeInfo.from_node(self._node.next)

    @property
    def prev(self):
        """
        Property to access the previous node.

        Returns:
        - NodeInfo: The previous node.
        """
        if self._node.prev.op == "root":
            return
        return NodeInfo.from_node(self._node.prev)

    # additional properties for detection
    @property
    def operator(self) -> Optional[Callable]:
        """
        Property to access the operator.

        Returns:
        - Optional[Callable]: The operator.
        """
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
        """
        Property to access the owning module.

        Returns:
        - Optional[str]: The owning module.
        """
        if self.opcode in ["call_module", "call_function"]:
            if self.meta.get("nn_module_stack"):
                nm = next(iter(self.meta["nn_module_stack"]))
                return nm
        return

    # convert data format
    def to_dict(self):
        """
        Converts the NodeInfo object to a dictionary.

        Returns:
        - dict: The dictionary representation of the NodeInfo object.
        """
        return {**asdict(self), "operator": self.operator}

    # [TODO] to_json for visualization
    def to_json_serializable(self):
        pass


class ModelArchitecture:
    """
    Represents the architecture of a model with methods for manipulating nodes.

    Attributes:
    - model: The model for which the architecture is defined.
    - tracer: A tracer to trace the model. By default, the  model is traced by `torch.fx.symbolic_trace`.
    """

    def __init__(self, model, tracer: Tracer|None=None):
        self.model = model
        self.tracer = tracer

        self._traced_model = self._trace(model)
        self._replacing = False
    
    def _trace(self, model):
        if self.tracer is None:
            return symbolic_trace(model)
        graph = self.tracer.trace(model)
        name = (
            model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
        )
        return GraphModule(self.tracer.root, graph, name)

    def list_nodes(self) -> List[NodeInfo]:
        """
        Lists all nodes in the model.

        Returns:
        - List[NodeInfo]: A list of NodeInfo objects representing the nodes in the model.
        """
        return [NodeInfo.from_node(n) for n in self._traced_model.graph.nodes]

    def get_node(self, name: str) -> NodeInfo:
        """
        Retrieves a node by its name.

        Args:
        - name (str): The name of the node to retrieve.

        Returns:
        - NodeInfo: The NodeInfo object corresponding to the specified node name.
        """
        for n in self._traced_model.graph.nodes:
            if n.name == name:
                return NodeInfo.from_node(n)
        return  # [TODO] Error for no result?

    def find_node(
        self,
        filter_func: Callable[[NodeInfo], bool],
        root: Optional[NodeInfo] = None,
        get_all: bool = False
    ) -> Union[NodeInfo, List[NodeInfo]]:
        """
        Finds a node based on a filtering function.

        Args:
            filter_func (Callable): The function used to filter nodes.
            root (Optional[NodeInfo]): The root node from which to start the search.
            get_all (bool): Whether to find all nodes matching the criteria.

        Returns:
            Union[NodeInfo, List[NodeInfo]]: The found node(s) or None if no node is found.
        """
        if root is None:
            # Take the first node
            root = NodeInfo.from_node(next(iter(self._traced_model.graph.nodes)))

        node = root
        nodes = []
        while node is not None:
            is_found = filter_func(node)
            if is_found:
                if not get_all:
                    return node
                nodes.append(node)
            node = node.next
        if len(nodes) == 0 and not get_all:
            return None
        return nodes

    def _validate_new_node(self, new_node: NodeInfo):
        if new_node.name is not None:
            exists = self.get_node(name=new_node.name)
            assert not exists, f"A node named {new_node.name} already exists."
        return True

    def _ensure_graph(self) -> None:
        self._traced_model.graph.lint()
        self._traced_model.recompile()

    def replace_node(self, node: NodeInfo, new_node: NodeInfo) -> NodeInfo:
        """
        Replaces a node in the model with a new node.

        Args:
        - node (NodeInfo): The node to replace.
        - new_node (NodeInfo): The new node to insert.

        Returns:
        - NodeInfo: The inserted node.
        """
        self._replacing = True
        self._validate_new_node(new_node)
        try:
            if new_node._mode == "from_module":
                new_node.name = f"{REPLACE_PREFIX}{node.name}"
            inserted = self.insert_node(new_node, base_node=node)
            self._traced_model.graph.erase_node(node._node)
            self._ensure_graph()
        finally:
            self._replacing = False
        return inserted

    def insert_node(self, new_node: NodeInfo, base_node: NodeInfo, before=False) -> NodeInfo:
        """
        Inserts a new node into the model.

        Args:
        - new_node (NodeInfo): The new node to insert.
        - base_node (NodeInfo): The node before which or after which to insert the new node.
        - before (bool): Whether to insert the new node before the base node.

        Returns:
        - NodeInfo: The inserted node.
        """
        self._validate_new_node(new_node)
        inserting = self._traced_model.graph.inserting_before if before else self._traced_model.graph.inserting_after
        if self._replacing:
            _inserted_args = base_node._node.args
            _inserted_kwargs = base_node._node.kwargs
            pass
        elif before:
            _inserted_args = tuple(
                arg for arg in base_node._node.args if isinstance(arg, Node))
            _inserted_kwargs = {
                kw: arg for kw, arg in base_node._node.kwargs.items() if isinstance(arg, Node)}
            _inserted_kwargs = {**_inserted_kwargs, **new_node._kwargs}
        else:
            _inserted_args = (base_node,)
            _inserted_kwargs = new_node._kwargs

        # [TODO] validation for new_node: if new_node._mode=="from_module" , new_node.name exists
        # insert
        with inserting(base_node._node):
            if new_node._mode == "from_module":
                self._traced_model.add_submodule(
                    new_node.name, new_node.operator)
                _inserted = self._traced_model.graph.call_module(
                    new_node.name,
                    _inserted_args,
                    _inserted_kwargs,
                )
            elif new_node._mode == "from_function":
                _inserted = self._traced_model.graph.call_function(
                    new_node.operator,
                    _inserted_args,
                    _inserted_kwargs,
                )

        if self._replacing or not before:
            base_node._node.replace_all_uses_with(_inserted)
            pass
        elif before:
            _not_inserted_args = [
                arg for arg in base_node._node.args if not isinstance(arg, Node)]
            base_node._node.args = tuple([_inserted] + _not_inserted_args)
            base_node._node.kwargs = {
                kw: arg for kw, arg in base_node._node.kwargs.items() if not isinstance(arg, Node)}
        self._ensure_graph()
        return NodeInfo.from_node(_inserted)

    def to_dict(self):
        """
        Converts the model architecture to a dictionary representation.

        Returns:
        - dict: A dictionary containing nodes and edges of the model architecture.
        """
        nodes = []
        edges = []
        for n in self.list_nodes():
            nodes.append(n.to_dict())
            edges += [{"source": n.name, "target": u.name} for u in n.users]
        return {"nodes": nodes, "edges": edges}
