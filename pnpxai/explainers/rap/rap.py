from typing import Dict, Type

import torch
from torch import nn, Tensor
from pnpxai.explainers.rap import rules
from pnpxai.explainers.utils.operation_graph import OperationGraph, OperationNode

SUPPORTED_MODULES: Dict[Type[nn.Module], Type[rules.RelProp]] = {
    nn.Sequential: rules.Sequential,
    nn.ReLU: rules.ReLU,
    nn.Dropout: rules.Dropout,
    nn.MaxPool2d: rules.MaxPool2d,
    nn.AdaptiveAvgPool2d: rules.AdaptiveAvgPool2d,
    nn.AvgPool2d: rules.AvgPool2d,
    nn.BatchNorm2d: rules.BatchNorm2d,
    nn.Linear: rules.Linear,
    nn.Conv2d: rules.Conv2d,
    nn.Flatten: rules.Flatten,
}
SUPPORTED_FUNCTIONS: Dict[callable, Type[rules.RelProp]] = {
    torch.add: rules.Add,
    torch.flatten: rules.Flatten,
}
SUPPORTED_BUILTINS: Dict[str, Type[rules.RelProp]] = {
    'add': rules.Add,
    'flatten': rules.Flatten,
}


class RelativeAttributePropagation():
    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = OperationGraph(model)

        self._assign_rules_and_hooks(self.graph.root)

    def _assign_rules_and_hooks(self, node: OperationNode):
        if node.is_module:
            layer = node.operator
            if type(layer) in SUPPORTED_MODULES and not (hasattr(layer, 'rule')):
                rule = SUPPORTED_MODULES[type(layer)]
                layer.rule: rules.RelProp = rule(layer)
                layer.register_forward_hook(layer.rule.forward_hook)

        for next_node in node.next_nodes:
            if not next_node.is_output:
                self._assign_rules_and_hooks(next_node)

    def get_node_rule(self, node: OperationNode) -> rules.RelProp:
        operator = node.operator
        rule = None
        if node.is_placeholder:
            rule = rules.RelProp()
        elif node.is_module and type(operator) in SUPPORTED_MODULES:
            rule = operator.rule
        elif node.is_function:
            if type(operator) in SUPPORTED_FUNCTIONS:
                rule = SUPPORTED_FUNCTIONS[type(operator)](operator)
            else:
                built_in_name = str(operator)[1:-1].split(' ')
                if len(built_in_name) >= 3 and built_in_name[2] in SUPPORTED_BUILTINS:
                    rule = SUPPORTED_BUILTINS[built_in_name[2]]()

        if rule is None:
            raise NotImplementedError(f'Unsupported node: {node}')

        return rule

    def relprop(self, r: Tensor) -> Tensor:
        stack = {node: [r] for node in self.graph.tail.prev_nodes}

        while len(stack) > 0:
            node, rs = stack.popitem()

            if len(rs) < len(node.users):
                preserved_node, preserved_rs = node, rs
                node, rs = stack.popitem()
                stack[preserved_node] = preserved_rs

            args_list = [
                prev_node.operator.rule.Y
                for prev_node in node.prev_nodes
                if prev_node.is_module
            ]
            args = args_list[0] if len(args_list) == 1 else args_list
            cur_r = sum(rs)

            rule = self.get_node_rule(node)
            r = rule.relprop(cur_r, args)

            if len(node.prev_nodes) == 0:
                continue

            if torch.is_tensor(r):
                r = [r]

            if len(r) != len(node.prev_nodes):
                print(node, len(r), len(node.prev_nodes))
                assert len(r) == len(node.prev_nodes)

            for prev_node, prev_r in zip(node.prev_nodes, r):
                stack.setdefault(prev_node, []).append(prev_r)

        return r
