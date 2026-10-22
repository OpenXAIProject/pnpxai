"""FX execution and feature-to-input Attribution Equilibrium propagation."""

import copy
import math
import operator
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch import Tensor, fx, nn
from torch.nn import functional as F

from pnpxai.explainers.ae.rule_map import (
    SUPPORTED_OPS,
    module_rule,
    reference_helper_rule,
)
from pnpxai.explainers.ae.rules.base import safe_divide


def _residual_block(module):
    return (
        type(module).__name__ in ("BasicBlock", "Bottleneck")
        and all(hasattr(module, name) for name in ("conv1", "bn1", "conv2", "bn2"))
    )


def _dense_block(module):
    return type(module).__name__ == "_DenseBlock"


class _Tracer(fx.Tracer):
    def is_leaf_module(self, module, module_qualified_name):
        return (
            _residual_block(module)
            or _dense_block(module)
            or module_rule(module) is not None
            or super().is_leaf_module(module, module_qualified_name)
        )


class _ResidualMain(nn.Module):
    """The main path used by reference ResNet BasicBlock/Bottleneck.hedge."""

    def __init__(self, block):
        super().__init__()
        layers = OrderedDict()
        count = 3 if hasattr(block, "conv3") else 2
        for index in range(1, count + 1):
            layers["conv{}".format(index)] = getattr(block, "conv{}".format(index))
            layers["bn{}".format(index)] = getattr(block, "bn{}".format(index))
            if index < count:
                layers["relu{}".format(index)] = getattr(
                    block, "relu{}".format(index), getattr(block, "relu", None)
                )
        self.layers = nn.Sequential(layers)

    def forward(self, inputs):
        return self.layers(inputs)


class AttributionEquilibrium(fx.Interpreter):
    """Cache a model forward pass and propagate AE from a chosen feature layer.

    ``layer`` may name a module (including a container) or an FX node. For
    standard backbones the default follows the reference feature layer; for
    other models it selects the last Conv2d. Only ancestors of that layer need
    propagation rules, so the classifier head is evaluated with normal autograd.
    """

    def __init__(self, model: nn.Module, layer: Optional[Union[str, nn.Module]] = None):
        self.original_model = model
        self.original_layer = layer
        tracer = _Tracer()
        traced = fx.GraphModule(model, tracer.trace(model))
        traced.eval()
        super().__init__(traced, garbage_collect_values=False)
        self._trace = traced
        self._nodes = tuple(traced.graph.nodes)
        self._residuals = {}
        self._dense_blocks = {}
        self._drop_dense_input_skip = False
        self.layer = self._resolve_layer(layer)
        self._seed_node = self._resolve_node(self.layer)
        self.reset()

    def _resolve_layer(self, layer):
        if isinstance(layer, nn.Module):
            for name, module in self.original_model.named_modules():
                if module is layer:
                    return name
            raise ValueError("The AE layer must belong to the explained model.")
        if layer is not None:
            if not isinstance(layer, str):
                raise TypeError("AE layer must be a module, module path, or FX node name.")
            return layer
        model = self.original_model
        if hasattr(model, "layer4"):
            return "layer4"
        if hasattr(model, "Mixed_7c"):
            return "Mixed_7c"
        if hasattr(model, "features"):
            features = model.features
            if type(model).__name__ == "VGG" and isinstance(features[-1], nn.MaxPool2d):
                return "features.{}".format(len(features) - 2)
            return "features"
        candidates = [
            node.target for node in self._nodes
            if node.op == "call_module"
            and isinstance(self._trace.get_submodule(node.target), nn.Conv2d)
        ]
        if not candidates:
            raise ValueError("AE requires a convolutional feature layer; specify layer explicitly.")
        return candidates[-1]

    def _resolve_node(self, layer):
        if layer == "__output__":
            output = next(node for node in self._nodes if node.op == "output")
            if isinstance(output.args[0], fx.Node):
                return output.args[0]
            raise ValueError("An AE block must return a single feature tensor.")
        candidates = []
        for node in self._nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            stack = node.meta.get("nn_module_stack", {})
            if (
                node.name == layer
                or (node.op == "call_module" and node.target == layer)
                or layer in stack
                or (node.op == "call_module" and node.target.startswith(layer + "."))
            ):
                candidates.append(node)
        if not candidates:
            raise ValueError(
                "AE layer {!r} is not present in the FX graph. For ResNet select "
                "a block output or stage rather than a layer inside a block.".format(layer)
            )
        return candidates[-1]

    def call_module(self, target, args, kwargs):
        module = self.fetch_attr(target)
        if getattr(module, "inplace", False):
            module = copy.copy(module)
            module.inplace = False
        return module(*args, **kwargs)

    def call_function(self, target, args, kwargs):
        if target in (F.relu, F.dropout):
            kwargs = dict(kwargs)
            index = 1 if target is F.relu else 3
            if len(args) > index:
                args = tuple(False if i == index else arg for i, arg in enumerate(args))
            else:
                kwargs["inplace"] = False
        return target(*args, **kwargs)

    def call_method(self, target, args, kwargs):
        if target == "relu_":
            target = "relu"
        return getattr(args[0], target)(*args[1:], **kwargs)

    def run_node(self, node):
        self._calls[node] = self.fetch_args_kwargs_from_env(node)
        result = super().run_node(node)
        self._results[node] = result
        return result

    def reset(self):
        self._results = {}
        self._calls = {}
        self.env = {}
        self.args_iter = iter(())

    def run(self, *args):
        self.reset()
        try:
            return super().run(*args)
        except Exception:
            self.reset()
            raise

    @staticmethod
    def _sum_relevance(left, right):
        if torch.is_tensor(left) and torch.is_tensor(right):
            return left + right
        if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
            if len(left) == len(right):
                return [AttributionEquilibrium._sum_relevance(a, b) for a, b in zip(left, right)]
        raise RuntimeError("Incompatible relevance maps at a graph branch.")

    def _block_relprop(self, node, rel, inputs, ratio):
        if node.target not in self._residuals:
            main = _ResidualMain(self._trace.get_submodule(node.target))
            self._residuals[node.target] = AttributionEquilibrium(main, layer="layers")
        method = self._residuals[node.target]
        try:
            method.run(inputs.detach())
            # Reference ResNet splits by two, propagates the main path, then
            # returns x1 + x1. Its computed downsample relevance is unused.
            return 2 * method.relprop(rel / 2, ratio=ratio)
        finally:
            method.reset()

    def _rule(self, node):
        if node.op == "call_module":
            module = self._trace.get_submodule(node.target)
            rule_type = module_rule(module)
            if rule_type is not None:
                return rule_type(module)
        else:
            rule_type = SUPPORTED_OPS.get(node.op, {}).get(node.target)
            if rule_type is not None:
                return rule_type()
        raise NotImplementedError(
            "AE has no propagation rule for node {!r} ({}: {}).".format(
                node.name, node.op, node.target
            )
        )

    def _dense_relprop(self, node, rel, inputs, ratio):
        if node.target not in self._dense_blocks:
            block = self._trace.get_submodule(node.target)
            method = AttributionEquilibrium(block, layer="__output__")
            # Reference _DenseBlock.hedge starts from the generated features
            # and returns norm1_rel. It omits first_rel[0], the direct output
            # concatenation contribution to the block's initial input.
            method._drop_dense_input_skip = True
            self._dense_blocks[node.target] = method
        method = self._dense_blocks[node.target]
        try:
            method.run(inputs.detach())
            return method.relprop(rel, ratio=ratio)
        finally:
            method.reset()

    def _node_relprop(self, node, rel, input_nodes, ratio):
        args, kwargs = self._calls[node]
        inputs = [self._results[parent] for parent in input_nodes]
        inputs = inputs[0] if len(inputs) == 1 else inputs
        if node.op == "call_module":
            module = self._trace.get_submodule(node.target)
            if _residual_block(module):
                return self._block_relprop(node, rel, inputs, ratio)
            if _dense_block(module):
                return self._dense_relprop(node, rel, inputs, ratio)
            helper = reference_helper_rule(module)
            if helper is not None and type(module).__name__ == "Add":
                args = tuple(args[0])
        return self._rule(node).relprop(
            rel, inputs, self._results[node], args, kwargs, ratio=ratio
        )

    def relprop(self, rel: Tensor, ratio: float = 2.0) -> Tensor:
        """Propagate a feature relevance map back to the first image input."""
        if self._seed_node not in self._results:
            raise RuntimeError("Call run() before AE relprop().")
        pending = {self._seed_node: rel.detach()}
        input_relevance = {}
        try:
            for node in reversed(self._nodes):
                if node not in pending:
                    continue
                relevance = pending.pop(node)
                if node.op == "placeholder":
                    input_relevance[node] = relevance
                    continue
                args, _ = self._calls[node]
                if (
                    node.op == "call_function" and node.target is operator.getitem
                    and isinstance(args[0], (tuple, list))
                ):
                    if not isinstance(args[1], int):
                        raise NotImplementedError("AE structured indexing requires an integer index.")
                    parent = node.args[0]
                    parts = [torch.zeros_like(value) for value in args[0]]
                    parts[args[1]] = relevance
                    contributions = [(parent, parts)]
                else:
                    input_nodes = [
                        parent for parent in node.all_input_nodes
                        if parent.op != "get_attr"
                        and torch.is_tensor(self._results[parent])
                        and self._results[parent].is_floating_point()
                    ]
                    if not input_nodes:
                        continue
                    propagated = self._node_relprop(node, relevance, input_nodes, ratio)
                    maps = [propagated] if torch.is_tensor(propagated) else propagated
                    if len(maps) != len(input_nodes):
                        raise RuntimeError("AE rule returned the wrong number of input maps at {}.".format(node.name))
                    if self._drop_dense_input_skip and node is self._seed_node:
                        maps = list(maps)
                        maps[0] = torch.zeros_like(maps[0])
                    contributions = list(zip(input_nodes, maps))
                for parent, value in contributions:
                    expected = self._results[parent]
                    if torch.is_tensor(value):
                        if value.shape != expected.shape:
                            raise RuntimeError("AE relevance shape mismatch at {}.".format(node.name))
                        value = value.detach()
                    pending[parent] = (
                        self._sum_relevance(pending[parent], value)
                        if parent in pending else value
                    )
            first_input = next(node for node in self._nodes if node.op == "placeholder")
            if first_input not in input_relevance:
                raise RuntimeError("The selected AE feature layer is not connected to the image input.")
            return input_relevance[first_input]
        finally:
            self.reset()

    @torch.enable_grad()
    def attribute(self, inputs: Tensor, targets=None, ratio: float = 2.0, additional_args=()):
        """Build the reference gradient-weighted activation and propagate it."""
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 1 <= ratio <= 2:
            raise ValueError("AE ratio must be a finite number between 1 and 2.")
        if not torch.is_tensor(inputs) or inputs.ndim != 4 or inputs.shape[0] == 0:
            raise ValueError("AE requires a nonempty batched NCHW image tensor.")
        if not inputs.is_floating_point():
            raise TypeError("AE image inputs must have a floating-point dtype.")
        try:
            output = self.run(inputs.detach().requires_grad_(), *additional_args)
            if not torch.is_tensor(output) or output.ndim != 2 or output.shape[0] != inputs.shape[0]:
                raise ValueError("AE requires classification logits shaped (batch, classes).")
            if targets is None:
                targets = output.argmax(dim=1)
            else:
                targets = torch.as_tensor(targets, device=output.device)
                if targets.is_floating_point() or targets.is_complex() or targets.dtype == torch.bool:
                    raise TypeError("AE targets must contain integer class indices.")
                if targets.ndim == 0:
                    targets = targets.expand(output.shape[0])
                if targets.shape != (output.shape[0],):
                    raise ValueError("AE requires one target class per input image.")
                targets = targets.long()
                if ((targets < 0) | (targets >= output.shape[1])).any():
                    raise ValueError("An AE target class is outside the model's class range.")
            features = self._results[self._seed_node]
            if not torch.is_tensor(features) or features.ndim != 4:
                raise ValueError("The selected AE layer must produce an NCHW feature tensor.")
            scores = output.gather(1, targets[:, None]).sum()
            gradients = torch.autograd.grad(scores, features)[0]
            activation = features.detach() * gradients.detach().mean((2, 3), keepdim=True)
            seed = safe_divide(activation, activation.sum().abs())
            return self.relprop(seed, ratio=ratio)
        finally:
            self.reset()
