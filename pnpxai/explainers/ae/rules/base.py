"""Common propagation rules adapted from Attribution Equilibrium's layers.py.

Rules receive intermediate tensors explicitly so they can be used by a graph
executor without replacing the model's layers or installing forward hooks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


_TensorOrTensors = Union[Tensor, Sequence[Tensor]]


def safe_divide(a: Tensor, b: Tensor) -> Tensor:
    """Divide elementwise, returning zero wherever the denominator is zero."""
    nonzero = b.ne(0)
    denominator = torch.where(nonzero, b, torch.ones_like(b))
    quotient = a / denominator
    return torch.where(nonzero, quotient, torch.zeros_like(quotient))


class RelProp:
    """Base AE rule; relevance passes through unchanged by default.

    ``ratio`` is accepted by every rule for a uniform propagation interface.
    Only rules that modulate positive and negative relevance use its value.
    """

    def __init__(self, module: Optional[nn.Module] = None):
        self.module = module

    def gradprop(
        self,
        Z: _TensorOrTensors,
        X: _TensorOrTensors,
        S: _TensorOrTensors,
    ) -> Tuple[Tensor, ...]:
        """Compute vector-Jacobian products without accumulating parameter grads."""
        return torch.autograd.grad(Z, X, S, retain_graph=True)

    def relprop(
        self,
        rel: _TensorOrTensors,
        inputs: Optional[_TensorOrTensors] = None,
        outputs: Optional[_TensorOrTensors] = None,
        args=None,
        kwargs=None,
        ratio: float = 2.0,
    ) -> _TensorOrTensors:
        """Propagate relevance through the graph executor's rule interface."""
        return rel


class RelPropSimple(RelProp):
    """AE's simple rule for pooling and other parameter-free operations.

    Recompute the operation on ``clamp(inputs, min=0) + 1e-9`` before applying
    ``inputs * grad(outputs, inputs, rel / outputs)``, as in the reference AE
    implementation. Saved outputs from the original forward pass are not used.
    Functional rules can override ``forward`` to replay their operation.
    """

    def forward(
        self, inputs: _TensorOrTensors, args=None, kwargs=None
    ) -> Tensor:
        """Replay a module with replacement inputs and its original arguments.

        ``args`` and ``kwargs`` are the original call arguments. This default
        handles a single tensor input; rules for multiple inputs override it.
        """
        if self.module is None:
            raise NotImplementedError(
                "Functional AE rules must implement forward()."
            )
        if not torch.is_tensor(inputs):
            raise NotImplementedError(
                "AE rules with multiple inputs must implement forward()."
            )
        call_args = tuple(args[1:]) if args else ()
        call_kwargs = dict(kwargs) if kwargs is not None else {}
        call_kwargs.pop("input", None)
        return self.module(inputs, *call_args, **call_kwargs)

    @torch.enable_grad()
    def backward(
        self,
        rel: Tensor,
        inputs: _TensorOrTensors,
        args=None,
        kwargs=None,
    ) -> _TensorOrTensors:
        """Propagate one relevance tensor through a locally recomputed graph."""
        single_input = torch.is_tensor(inputs)
        input_tensors = (inputs,) if single_input else tuple(inputs)
        positive_inputs = tuple(
            x.detach().clamp(min=0).add(1e-9).requires_grad_()
            for x in input_tensors
        )
        local_inputs = positive_inputs[0] if single_input else positive_inputs
        local_outputs = self.forward(local_inputs, args, kwargs)
        scales = safe_divide(rel, local_outputs)
        gradients = self.gradprop(local_outputs, positive_inputs, scales)
        propagated = [x * grad for x, grad in zip(positive_inputs, gradients)]
        return propagated[0] if single_input else propagated

    def relprop(
        self,
        rel: _TensorOrTensors,
        inputs: _TensorOrTensors,
        outputs: Optional[_TensorOrTensors] = None,
        args=None,
        kwargs=None,
        ratio: float = 2.0,
    ) -> _TensorOrTensors:
        """Propagate one map or a sequence of maps independently.

        A sequence denotes separate relevance maps for the same operation,
        matching the reference implementation's behavior.
        """
        if torch.is_tensor(rel):
            return self.backward(rel, inputs, args, kwargs)
        return [self.backward(value, inputs, args, kwargs) for value in rel]


class RelPropSeparate(RelProp, ABC):
    """Shared signed-relevance calculations for Attribution Equilibrium.

    Subclasses implement ``forward`` and ``agg_dims``, and provide a layer's
    ``relprop`` entry point. That entry point selects special propagation paths
    and prepares positive inputs and absolute weights for ``backward``.

    The reference Conv2d rule also adds activation-mask contributions and uses
    special paths for 1x1 kernels and the input layer. Those layer-specific
    calculations belong in ``modules.py``.
    """

    @property
    @abstractmethod
    def agg_dims(self) -> Union[int, Sequence[int]]:
        """Dimensions over which relevance is summed for shift correction."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor, weights: Tensor) -> Tensor:
        """Apply the layer operation with explicit inputs and weights."""
        raise NotImplementedError

    def normalize_rel(self, rel: Tensor) -> Tuple[Tensor, Tensor]:
        """Normalize positive and negative relevance to sums of +1 and -1.

        As in the reference AE implementation, normalization uses the total
        over the relevance tensor. An absent sign is returned as zeros.
        ``ratio`` is applied afterwards to the positive component.
        """
        pos_rel = rel.clamp(min=0)
        neg_rel = rel.clamp(max=0)
        return (
            safe_divide(pos_rel, pos_rel.sum()),
            -safe_divide(neg_rel, neg_rel.sum()),
        )

    def att_prop(
        self,
        pos_rel: Tensor,
        neg_rel: Tensor,
        pos_out: Tensor,
        neg_out: Tensor,
        inputs: Tensor,
    ) -> Tensor:
        """Propagate AE's positive and negative relevance independently."""
        pos_scale = safe_divide(pos_rel, pos_out)
        neg_scale = safe_divide(neg_rel, neg_out)
        pos_grad = self.gradprop(pos_out, inputs, pos_scale)[0]
        neg_grad = self.gradprop(neg_out, inputs, neg_scale)[0]
        return inputs * pos_grad + inputs * neg_grad

    def shift_rel(self, rel: Tensor, rel_diff: Tensor) -> Tensor:
        """Subtract an equal share of ``rel_diff`` at every nonzero location."""
        nonzero = rel.ne(0)
        count = nonzero.sum(dim=self.agg_dims, keepdim=True)
        shift = safe_divide(rel_diff, count) * nonzero
        return rel - shift

    @torch.enable_grad()
    def backward(
        self,
        rel: Tensor,
        pos_x: Tensor,
        abs_w: Tensor,
        ratio: float = 2.0,
    ) -> Tensor:
        """Compute AE's ratio-modulated attribution and shift correction.

        ``pos_x`` is the nonnegative input and ``abs_w`` the absolute weight.
        The returned value is the common attribution component; a layer rule
        adds its activation-mask contributions separately.
        """
        inputs = pos_x.detach().requires_grad_()
        pos_rel, neg_rel = self.normalize_rel(rel)
        scaled_pos_rel = pos_rel * ratio

        outputs = self.forward(inputs, abs_w)
        pos_out = outputs * scaled_pos_rel.ne(0)
        neg_out = outputs * neg_rel.ne(0)
        propagated = self.att_prop(
            scaled_pos_rel, neg_rel, pos_out, neg_out, inputs
        )

        # AE subtracts the normalized positive mass, rather than the difference
        # between outgoing and incoming sums used by RAP's propagation rule.
        shift_mass = safe_divide(scaled_pos_rel, scaled_pos_rel.sum()).sum(
            dim=self.agg_dims, keepdim=True
        )
        return self.shift_rel(propagated, shift_mass)
