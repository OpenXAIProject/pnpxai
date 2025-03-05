from collections import OrderedDict, defaultdict
from typing import Dict, Sequence, Callable, Optional, Tuple, Set

import warnings
import torch
from torch import nn, Tensor, fx
from pnpxai.explainers.rap import rules
from pnpxai.explainers.rap.rule_map import SUPPORTED_OPS
from pnpxai.messages import get_message
from pnpxai.utils import flatten, map_recursive


class RelativeAttributePropagation():

    
    def __init__(self, model: nn.Module):
        self._trace = fx.symbolic_trace(model)
        self._trace.eval()
        self.reset()

    def _backprop_unused(self, node: fx.Node, unused: Set[str]) -> Set[str]:
        unused.add(node.name)
        for arg in node.all_input_nodes:
            if all([user.name in unused for user in arg.users]):
                unused = unused.union(self._backprop_unused(arg, unused))
        return unused

    def _check_unused_by_result(self, node: fx.Node, result) -> bool:
        return node.op != 'output' and (len(node.users) == 0 or not any([torch.is_tensor(res) for res in flatten(result)]))

    def _load_args(self, args, modifier: Optional[Callable] = None):
        modifier = modifier if modifier is not None else (lambda x: x)
        return fx.map_arg(args, lambda node: modifier(self._results[node.name]))

    def _fetch_attr(self, target: str):
        target_atoms = target.split('.')
        attr_itr = self._trace
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced non-existent target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def _step_node(self, node: fx.Node, arg_modifier: Optional[Callable] = None) -> Tuple[Tensor, Tuple[Sequence[Tensor], Dict[str, Tensor]]]:
        result = None
        args = self._load_args(node.args, arg_modifier)
        kwargs = self._load_args(node.kwargs, arg_modifier)

        if node.op == 'get_attr':
            result = self._fetch_attr(node.target)
        elif node.op == 'call_function':
            result = node.target(*args, **kwargs)
        elif node.op == 'call_method':
            result = getattr(args[0], node.target)(*args[1:], **kwargs)
        elif node.op == 'call_module':
            result = self._fetch_attr(node.target)(*args, **kwargs)
        elif node.op == 'output':
            result = args[0]

        if self._check_unused_by_result(node, result):
            self._unused_nodes = self._unused_nodes.union(
                self._backprop_unused(node, self._unused_nodes))

        return result, (args, kwargs)

    def _marked_unused(self, node: fx.Node) -> bool:
        return node.op != 'output' and (node.name in self._unused_nodes)

    def run(self, *args):
        """
        Method to preserve all intermediary states
        """
        args_iter = iter(args)
        graph = self._trace.graph

        for node in graph.nodes:
            node: fx.Node
            if node.op == 'placeholder':
                result = next(args_iter)
            else:
                result, _ = self._step_node(node)

            self._results[node.name] = result

        return self._results['output']

    def _get_init_relprop_stack(self, rel: Sequence[Tensor]) -> Dict[fx.Node, None]:
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

    def _all_have_grad(self, args) -> bool:
        return all([
            arg.requires_grad and arg.grad_fn
            for arg in flatten(args) if torch.is_tensor(arg)
        ])

    def _node_relprop(self, node: fx.Node):
        def enable_grad(x): return x.clone().requires_grad_()

        args, kwargs = self._load_args(node.args), self._load_args(node.kwargs)
        inputs = self._load_args(node.all_input_nodes)
        outputs = self._results[node.name]
        if not self._all_have_grad(inputs):
            outputs, (args, kwargs) = self._step_node(
                node, lambda result: map_recursive(result, enable_grad)
            )
            inputs = [args, kwargs]

        inputs = [arg for arg in flatten(inputs) if torch.is_tensor(arg)]

        if not torch.is_tensor(inputs) and len(inputs) == 1:
            inputs = inputs[0]

        rel = [
            self._relprops[node.name][user.name]
            for user in node.users.keys() if not self._marked_unused(user)
        ]

        if len(node.users) > 1:
            rel = sum(rel)
        elif len(rel) == 1:
            rel = rel[0]

        rule = None
        if node.op == 'placeholder' or node.op == 'get_attr':
            rule = rules.RelProp()
        elif node.op == 'call_module':
            target = self._fetch_attr(node.target)
            rule = SUPPORTED_OPS[node.op][type(target)](target)
        elif node.op == 'call_function' or node.op == 'call_method':
            rule = SUPPORTED_OPS[node.op][node.target]()

        if rule is None:
            raise NotImplementedError(
                f"RelProp rule for node {node.name} is not implemented"
            )
        rel = rule.relprop(rel, inputs, outputs, args, kwargs)

        return rel

    def _node_has_all_users_relprops(self, node: fx.Node) -> bool:
        return all([
            self._relprops.get(node.name, {}).get(user.name, None) is not None
            for user in node.users.keys() if not self._marked_unused(user)
        ])
    
    def reset(self):
        self._results: Dict[str, fx.Node] = {}
        self._inputs: Dict[str, fx.Node] = defaultdict(tuple)
        self._relprops: Dict[str, Dict[str, Tensor]] = defaultdict(dict)
        # Solves bottleneck, when module has multiple outputs, some of which are unused
        self._unused_nodes = set()


    def relprop(self, r: Sequence[Tensor]) -> Tensor:
        queue = self._get_init_relprop_stack(r)
        outputs = []
        while len(queue) > 0:
            node = queue.popitem(last=False)[0]
            if self._marked_unused(node):
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

            if node.op == 'placeholder':
                outputs.append(r)

            del self._relprops[node.name]

        self.reset()

        if len(outputs) == 1:
            return outputs[0]
        
        return outputs
