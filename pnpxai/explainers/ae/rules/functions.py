"""AE function rules and tensor-layout adapters for graph propagation.

ReLU, Add, Cat and Clone follow the reference AE rules in modules/layers.py.
Layout rules route relevance back to the corresponding input positions.
"""

from typing import Sequence, Tuple

import torch
from torch import Tensor

from pnpxai.explainers.ae.rules.base import (
    RelProp,
    RelPropSimple,
    _TensorOrTensors,
    safe_divide,
)


def _tensor_inputs(inputs: _TensorOrTensors) -> Tuple[Tensor, ...]:
    tensors = (inputs,) if torch.is_tensor(inputs) else tuple(inputs)
    if not tensors or not all(torch.is_tensor(x) for x in tensors):
        raise ValueError("AE function rules require tensor inputs.")
    return tensors


def _single_input(inputs: _TensorOrTensors) -> Tensor:
    tensors = _tensor_inputs(inputs)
    if len(tensors) != 1:
        raise ValueError("This AE function rule requires one tensor input.")
    return tensors[0]


def _argument(name, index, args=None, kwargs=None, default=None):
    if kwargs is not None and name in kwargs:
        return kwargs[name]
    return args[index] if args is not None and len(args) > index else default


def _route_branches(
    relevance: Sequence[Tensor],
    operands: Sequence[Tensor],
    inputs: _TensorOrTensors,
) -> _TensorOrTensors:
    """Return one map per graph input, merging repeated operand occurrences."""
    tensors = _tensor_inputs(inputs)
    if len(tensors) == len(operands) and all(
        x is operand for x, operand in zip(tensors, operands)
    ):
        result = list(relevance)
    else:
        result = []
        for x in tensors:
            matches = [r for operand, r in zip(operands, relevance) if operand is x]
            if not matches:
                raise ValueError("Graph inputs must match the operation's operands.")
            result.append(sum(matches[1:], matches[0]))
    return result[0] if torch.is_tensor(inputs) else result


class ReLU(RelProp):
    """Pass relevance through, as in the reference AE ReLU rule."""


class Add(RelPropSimple):
    """Distribute relevance using AE's absolute branch-mass weighting.

    This ports Add.hedge for two tensor operands with alpha=1. Global sums
    follow the reference implementation's normalization convention.
    """

    @torch.enable_grad()
    def backward(self, rel: Tensor, inputs: _TensorOrTensors, args=None, kwargs=None):
        tensors = _tensor_inputs(inputs)
        left = _argument("input", 0, args, kwargs, tensors[0])
        right = _argument(
            "other", 1, args, kwargs, tensors[1] if len(tensors) > 1 else None
        )
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise NotImplementedError("The AE Add rule requires two tensor operands.")
        if _argument("alpha", 2, args, kwargs, 1) != 1:
            raise NotImplementedError("The reference AE Add rule supports alpha=1.")

        local = tuple(x.detach().requires_grad_() for x in (left, right))
        output = torch.add(*local)
        gradients = self.gradprop(output, local, safe_divide(rel, output))
        branches = [x * grad for x, grad in zip(local, gradients)]
        masses = [branch.sum() for branch in branches]
        total_mass = masses[0].abs() + masses[1].abs()
        propagated = [
            branch * safe_divide(
                safe_divide(mass.abs(), total_mass) * rel.sum(), mass
            )
            for branch, mass in zip(branches, masses)
        ]
        return _route_branches(propagated, (left, right), inputs)


class Cat(RelPropSimple):
    """Propagate through concatenation using the reference AE input-times-grad rule."""

    @torch.enable_grad()
    def backward(self, rel: Tensor, inputs: _TensorOrTensors, args=None, kwargs=None):
        operands = _argument("tensors", 0, args, kwargs, _tensor_inputs(inputs))
        operands = tuple(operands)
        _tensor_inputs(operands)
        dim = _argument("dim", 1, args, kwargs, 0)
        local = tuple(x.detach().requires_grad_() for x in operands)
        output = torch.cat(local, dim=dim)
        gradients = self.gradprop(output, local, safe_divide(rel, output))
        propagated = [x * grad for x, grad in zip(local, gradients)]
        return _route_branches(propagated, operands, inputs)


class Clone(RelProp):
    """Merge duplicate branches with the reference AE sign-product rule.

    ``rel`` contains one relevance map per branch. This is AE's multi-output
    Clone operation; a single-output torch.clone only needs an identity rule.
    """

    def relprop(
        self, rel, inputs, outputs=None, args=None, kwargs=None, ratio: float = 2.0
    ) -> Tensor:
        if torch.is_tensor(rel) or not rel:
            raise ValueError("AE Clone requires a nonempty sequence of branch maps.")
        count = _argument("num", 1, args, kwargs, len(rel))
        if count != len(rel) or (outputs is not None and len(outputs) != len(rel)):
            raise ValueError("AE Clone requires one relevance map per output branch.")
        x = _single_input(inputs)
        if any(not torch.is_tensor(r) or r.shape != x.shape for r in rel):
            raise ValueError("AE Clone branch maps must match the input shape.")
        pos_grad = sum(safe_divide(r.clamp(min=0), x) for r in rel)
        neg_grad = sum(safe_divide(r.clamp(max=0), x) for r in rel)
        return x * (pos_grad * neg_grad)


class Flatten(RelProp):
    """Restore the input layout, independently for each relevance map."""

    def backward(self, rel: Tensor, inputs: Tensor, args=None, kwargs=None) -> Tensor:
        return rel.reshape(inputs.shape)

    def relprop(
        self, rel, inputs, outputs=None, args=None, kwargs=None, ratio: float = 2.0
    ) -> _TensorOrTensors:
        x = _single_input(inputs)
        if torch.is_tensor(rel):
            return self.backward(rel, x, args, kwargs)
        return [self.backward(r, x, args, kwargs) for r in rel]


class Reshape(Flatten):
    """Undo reshape by restoring the input shape."""


class View(Flatten):
    """Undo view, including relevance maps that are noncontiguous."""


class Squeeze(Flatten):
    """Restore dimensions removed by squeeze."""


class Unsqueeze(Flatten):
    """Remove the dimension inserted by unsqueeze."""


class Permute(Flatten):
    """Apply the inverse of the forward dimension permutation."""

    def backward(self, rel: Tensor, inputs: Tensor, args=None, kwargs=None) -> Tensor:
        dims = (kwargs or {}).get("dims")
        if dims is None and args is not None:
            dims = args[1] if len(args) == 2 and isinstance(args[1], (tuple, list)) else args[1:]
        if dims is None:
            raise ValueError("AE Permute requires the forward permutation dimensions.")
        dims = tuple(dim % inputs.ndim for dim in dims)
        if sorted(dims) != list(range(inputs.ndim)):
            raise ValueError("AE Permute requires a permutation of all input dimensions.")
        inverse = tuple(dims.index(dim) for dim in range(inputs.ndim))
        return rel.permute(inverse)


class Transpose(Flatten):
    """Undo a transpose using PyTorch's dim0 and dim1 call arguments."""

    def backward(self, rel: Tensor, inputs: Tensor, args=None, kwargs=None) -> Tensor:
        dim0 = _argument("dim0", 1, args, kwargs)
        dim1 = _argument("dim1", 2, args, kwargs)
        if dim0 is None or dim1 is None:
            raise ValueError("AE Transpose requires dim0 and dim1.")
        return rel.transpose(dim0, dim1)


class GetItem(Flatten):
    """Scatter relevance back through tensor indexing, accumulating repeated indices."""

    @torch.enable_grad()
    def backward(self, rel: Tensor, inputs: Tensor, args=None, kwargs=None) -> Tensor:
        if args is None or len(args) < 2:
            raise ValueError("AE GetItem requires the original tensor and index arguments.")
        if not torch.is_tensor(args[0]):
            raise NotImplementedError("AE GetItem currently supports tensor indexing.")
        template = torch.zeros_like(inputs, requires_grad=True)
        selected = template[args[1]]
        return self.gradprop(selected, template, rel)[0]


class GetAttr(RelProp):
    """Handle graph attribute access; shape metadata carries no relevance."""


class Contiguous(RelProp):
    """Pass relevance through a memory-layout-only operation."""
