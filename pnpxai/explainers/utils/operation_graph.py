from torch import fx, nn
from typing import Optional, List, Union, Callable, Dict


def get_model_operator(model: nn.Module, operator_name: str):
    operator_path = operator_name.split('.')
    for block in operator_path:
        model = model._modules[block]
    return model


class OperationNode:
    def __init__(
        self,
        node: fx.Node,
        operator: Optional[Union[nn.Module, Callable]] = None,
        next_nodes: Optional[List["OperationNode"]] = None,
        prev_nodes: Optional[List["OperationNode"]] = None,
    ):
        self._node = node
        self.operator = operator
        self._next_nodes = next_nodes or []
        self._prev_nodes = prev_nodes or []

    def add_next_node(self, node: "OperationNode"):
        self._next_nodes.append(node)

    def add_prev_node(self, node: "OperationNode"):
        self._prev_nodes.append(node)

    def set_operator_from_model(self, model: nn.Module):
        if self.is_module or self.is_function:
            target = self._node.target
            self.operator = get_model_operator(model, target) \
                if isinstance(target, str) else target

    @property
    def name(self):
        return self._node.name

    @property
    def users(self):
        return self._node.users

    @property
    def next_nodes(self):
        return self._next_nodes

    @property
    def prev_nodes(self):
        return self._prev_nodes

    @property
    def is_module(self) -> bool:
        return self._node.op == 'call_module'

    @property
    def is_function(self) -> bool:
        return self._node.op == 'call_function'

    @property
    def is_output(self) -> bool:
        return self._node.op == 'output'

    @property
    def is_placeholder(self) -> bool:
        return self._node.op == 'placeholder'

    def __repr__(self):
        return f'OperationNode({self._node.name})'


class OperationGraph:
    def __init__(self, model: nn.Module):
        self.root = None
        self.tail = None
        self._nodes_map: Dict[str, OperationNode] = {}
        self._model = model

        self._build()

    def _build(self):
        graph = fx.symbolic_trace(self._model).graph
        graph_nodes_map = {}

        for node in graph.nodes:
            graph_nodes_map[node.name] = node

        for node in graph.nodes:
            operation_node = OperationNode(node)
            operation_node.set_operator_from_model(self._model)
            self._nodes_map[node.name] = operation_node
            prev_nodes = node.args

            for prev_node in prev_nodes:
                if isinstance(prev_node, fx.Node):
                    prev_op_node = self._nodes_map[prev_node.name]
                    operation_node.add_prev_node(prev_op_node)
                    prev_op_node.add_next_node(operation_node)

            if node.op == 'placeholder':
                self.root = operation_node

            if node.op == 'output':
                self.tail = operation_node

    def pprint(self):
        def _pprint(node: OperationNode, indent: int = 0):
            print(' ' * indent, node._node.name)
            for next_node in node._next_nodes:
                if not next_node.is_output:
                    _pprint(next_node, indent + 1)

        _pprint(self.root)
