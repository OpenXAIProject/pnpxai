"""Layer rules adapted from Attribution Equilibrium's modules/layers.py."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pnpxai.explainers.ae.rules.base import (
    RelProp,
    RelPropSeparate,
    RelPropSimple,
    safe_divide,
)
from pnpxai.explainers.ae.rules.functions import _single_input


class Dropout(RelProp):
    """Pass relevance through, as in the reference AE Dropout.hedge."""


class _Pool2d(RelPropSimple):
    """Replay either a pooling module or its functional equivalent."""

    def forward(self, inputs, args=None, kwargs=None) -> Tensor:
        if self.module is not None:
            output = super().forward(inputs, args, kwargs)
        else:
            call_args = tuple(args[1:]) if args else ()
            call_kwargs = dict(kwargs) if kwargs is not None else {}
            call_kwargs.pop("input", None)
            output = self.function(inputs, *call_args, **call_kwargs)
        if not torch.is_tensor(output):
            raise NotImplementedError(
                "AE pooling requires a tensor output; use return_indices=False."
            )
        return output


class MaxPool2d(_Pool2d):
    """AE's positive-input simple propagation for 2D max pooling."""

    function = staticmethod(F.max_pool2d)


class AvgPool2d(_Pool2d):
    """AE's positive-input simple propagation for 2D average pooling."""

    function = staticmethod(F.avg_pool2d)


class AdaptiveAvgPool2d(_Pool2d):
    """AE's positive-input simple propagation for adaptive average pooling."""

    function = staticmethod(F.adaptive_avg_pool2d)


class BatchNorm2d(RelProp):
    """Apply the reference AE scale rule using stored running statistics.

    The reference computes scale with sqrt(running_var ** 2 + eps). Preserve
    that expression here; the normalization bias is not used for propagation.
    """

    def __init__(self, module: nn.BatchNorm2d):
        if not isinstance(module, nn.BatchNorm2d):
            raise TypeError("AE BatchNorm2d requires an nn.BatchNorm2d module.")
        if module.running_var is None:
            raise NotImplementedError("AE BatchNorm2d requires running statistics.")
        super().__init__(module)

    def relprop(
        self, rel, inputs, outputs=None, args=None, kwargs=None, ratio: float = 2.0
    ):
        x = _single_input(inputs)
        if x.ndim != 4:
            raise ValueError("AE BatchNorm2d requires batched NCHW input.")
        variance = self.module.running_var.detach().view(1, -1, 1, 1)
        weight = self.module.weight
        weight = (
            torch.ones_like(variance)
            if weight is None else weight.detach().view(1, -1, 1, 1)
        )
        scale = weight / (variance.square() + self.module.eps).sqrt()

        def propagate(value):
            return x * (safe_divide(value, x * scale) * scale)

        return propagate(rel) if torch.is_tensor(rel) else [propagate(r) for r in rel]


class Linear(RelProp):
    """Preserve the identity rule inherited by the reference AE Linear.hedge.

    Reference AE starts propagation at a convolutional feature map, below the
    classifier. Linear's separate alpha-beta relprop is not used by that path.
    This identity rule does not map a classifier output back to its input size.
    """


class Conv2d(RelPropSeparate):
    """AE propagation for a convolutional feature map.

    By default, three input channels select the reference input-layer rule.
    A graph executor can override ``input_layer`` when the actual input layer
    is known. The reference's special kernel path tests kernel height == 1,
    including rectangular 1-by-k kernels.
    """

    def __init__(self, module: nn.Conv2d, input_layer: Optional[bool] = None):
        if not isinstance(module, nn.Conv2d):
            raise TypeError("AE Conv2d requires an nn.Conv2d module.")
        if module.padding_mode != "zeros":
            raise NotImplementedError("AE Conv2d currently supports zero padding.")
        super().__init__(module)
        self.input_layer = input_layer

    @property
    def agg_dims(self):
        return (1, 2, 3)

    def forward(self, inputs: Tensor, weights: Tensor) -> Tensor:
        """Recompute a bias-free convolution with the original layer geometry."""
        return F.conv2d(
            inputs,
            weights,
            bias=None,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups,
        )

    @torch.enable_grad()
    def activation_prop(self, rel: Tensor, inputs: Tensor, abs_w: Tensor) -> Tensor:
        """Compute AE's positive and nonpositive activation-mask contributions."""
        pos_rel, neg_rel = self.normalize_rel(rel)
        pos_mask = inputs.gt(0).to(inputs.dtype).requires_grad_()
        neg_mask = inputs.le(0).to(inputs.dtype).requires_grad_()
        pos_out = self.forward(pos_mask, abs_w) * rel.ge(0)
        neg_out = self.forward(neg_mask, abs_w) * rel.le(0)
        pos_grad = self.gradprop(pos_out, pos_mask, safe_divide(pos_rel, pos_out))[0]
        neg_grad = self.gradprop(neg_out, neg_mask, safe_divide(neg_rel, neg_out))[0]
        return pos_mask * pos_grad + neg_mask * neg_grad

    @torch.enable_grad()
    def final_backward(self, rel: Tensor, inputs: Tensor) -> Tensor:
        """Propagate at the input using AE's per-sample lower and upper bounds.

        Differentiate independent local tensors to obtain the convolution
        adjoints. This also restores exact input shapes for strided kernels.
        The bounds are held fixed during these local gradient computations.
        """
        raw = inputs.detach()
        x = raw.requires_grad_()
        low = raw.amin(dim=self.agg_dims, keepdim=True)
        high = raw.amax(dim=self.agg_dims, keepdim=True)
        low = low.expand_as(raw).detach().requires_grad_()
        high = high.expand_as(raw).detach().requires_grad_()
        weight = self.module.weight.detach()
        norm_out = (
            self.forward(x, weight)
            - self.forward(low, weight.clamp(min=0))
            - self.forward(high, weight.clamp(max=0))
        )
        scale = safe_divide(rel, norm_out)
        grad_x, grad_low, grad_high = self.gradprop(norm_out, (x, low, high), scale)
        return x * grad_x + low * grad_low + high * grad_high

    @torch.enable_grad()
    def _propagate(self, rel: Tensor, inputs: Tensor, ratio: float) -> Tensor:
        is_input_layer = (
            inputs.shape[1] == 3 if self.input_layer is None else self.input_layer
        )
        if is_input_layer:
            return self.final_backward(rel, inputs)

        raw = inputs.detach()
        pos_x = raw.clamp(min=0).requires_grad_()
        abs_w = self.module.weight.detach().abs()
        if self.module.kernel_size[0] == 1:
            # The reference's height-one path uses unscaled signed relevance
            # and omits shift correction, so its result is independent of ratio.
            pos_rel, neg_rel = self.normalize_rel(rel)
            output = self.forward(pos_x, abs_w)
            common = self.att_prop(
                pos_rel, neg_rel,
                output * pos_rel.ne(0), output * neg_rel.ne(0), pos_x,
            )
        else:
            common = self.backward(rel, pos_x, abs_w, ratio=ratio)
        return common + self.activation_prop(rel, raw, abs_w)

    def relprop(
        self, rel, inputs, outputs=None, args=None, kwargs=None, ratio: float = 2.0
    ):
        x = _single_input(inputs)
        if x.ndim != 4:
            raise ValueError("AE Conv2d requires batched NCHW input.")
        if torch.is_tensor(rel):
            return self._propagate(rel, x, ratio)
        return [self._propagate(r, x, ratio) for r in rel]
