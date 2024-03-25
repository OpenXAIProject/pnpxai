from collections import OrderedDict, defaultdict
from typing import Dict, Sequence, Callable, Optional

import warnings
import torch
from torch import nn, Tensor, fx
from pnpxai.explainers.rap import rules
from pnpxai.explainers.rap.rule_map import SUPPORTED_OPS
from pnpxai.messages import get_message


class RelativeAttributePropagation():
    def __init__(self, model: nn.Module):
        self._trace = fx.symbolic_trace(model)
        self._trace.eval()
        self._results: Dict[str, fx.Node] = {}
        self._inputs: Dict[str, fx.Node] = defaultdict(tuple)
        self._relprops: Dict[str, Dict[str, Tensor]] = defaultdict(dict)

    def _load_args(self, args):
        return fx.graph.map_arg(args, lambda node: self._results[node.name])

    def _fetch_attr(self, target: str):
        target_atoms = target.split('.')
        attr_itr = self._trace
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced non-existent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def _step_node(self, node: fx.Node, args_iter: Tensor):
        result = None
        if node.op == 'placeholder':
            result = next(args_iter)
        elif node.op == 'get_attr':
            result = self._fetch_attr(node.target)
        elif node.op == 'call_function':
            result = node.target(
                *self._load_args(node.args),
                **self._load_args(node.kwargs)
            )
        elif node.op == 'call_method':
            self_obj, *args = self._load_args(node.args)
            kwargs = self._load_args(node.kwargs)
            result = getattr(self_obj, node.target)(*args, **kwargs)
        elif node.op == 'call_module':
            result = self._fetch_attr(node.target)(
                *self._load_args(node.args),
                **self._load_args(node.kwargs)
            )
        elif node.op == 'output':
            result = self._load_args(node.args)[0]
        return result

    def run(self, *args):
        """
        Method to preserve all intermediary states
        """
        args_iter = iter(args)
        graph = self._trace.graph

        for node in graph.nodes:
            node: fx.Node
            result = self._step_node(node, args_iter)
            self._results[node.name] = result
        return self._results['output']

    def _get_init_relprop_stack(self, rel: Sequence[Tensor]) -> OrderedDict[fx.Node, None]:
        tail = self._trace.graph._root
        while tail.op != 'output':
            tail = tail.next

        stack = OrderedDict({node: None for node in tail.all_input_nodes})
        for node in stack:
            self._relprops[node.name][tail.name] = rel
        return stack

    def _get_node_rule(self, node: fx.Node) -> Optional[rules.RelProp]:
        rule = None
        if node.op == 'placeholder':
            rule = rules.RelProp()
        elif node.op == 'get_attr' or node.op == 'call_module':
            target = self._fetch_attr(node.target)
            rule = SUPPORTED_OPS[node.op][type(target)](target)
        elif node.op == 'call_function' or node.op == 'call_method':
            rule = SUPPORTED_OPS[node.op][node.target]()

        return rule

    def _node_relprop(self, node: fx.Node):
        args = self._load_args(node.all_input_nodes)
        if not torch.is_tensor(args) and len(args) == 1:
            args = args[0]

        outputs = self._results[node.name]

        rel = [
            self._relprops[node.name][user.name]
            for user in node.users.keys()
        ]

        if len(node.users) > 1:
            rel = sum(rel)
        elif len(rel) == 1:
            rel = rel[0]

        rule = None
        if node.op == 'placeholder':
            rule = rules.RelProp()
        elif node.op == 'get_attr' or node.op == 'call_module':
            target = self._fetch_attr(node.target)
            rule = SUPPORTED_OPS[node.op][type(target)](target)
        elif node.op == 'call_function' or node.op == 'call_method':
            rule = SUPPORTED_OPS[node.op][node.target]()

        if rule is None:
            raise NotImplementedError(
                f"RelProp rule for node {node.name} is not implemented"
            )

        # TODO: Suggest better solution for arguments without gradient function
        if all([arg.requires_grad for arg in self._load_args(node.all_input_nodes)]):
            rel = rule.relprop(rel, args, outputs, node.args, node.kwargs)

        return rel

    def _node_has_all_users_relprops(self, node: fx.Node) -> bool:
        return all([
            self._relprops.get(node.name, {}).get(user.name, None) is not None
            for user in node.users.keys()
        ])

    def relprop(self, r: Sequence[Tensor]) -> Tensor:
        queue = self._get_init_relprop_stack(r)

        while len(queue) > 0:
            node = queue.popitem(last=False)[0]
            if len(node.users) == 0:
                continue

            if not self._node_has_all_users_relprops(node):
                queue[node] = None
                continue

            try:
                r = self._node_relprop(node)
            except Exception as e:
                warnings.warn(get_message(
                    'explainer.rap.errors.node', node=node.name))
                raise e

            for i, arg in enumerate(node.all_input_nodes):
                self._relprops[arg.name][node.name] = r if torch.is_tensor(
                    r) else r[i]
                queue[arg] = None

            del self._relprops[node.name]

        return r
