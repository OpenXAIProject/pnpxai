# Copyright (c) Karl Schulz, Leon Sixt
#
# All rights reserved.
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Gaussian CoIBA primitives, following the original IBA estimator/KL math."""

from contextlib import contextmanager
import hashlib
from math import isqrt

import torch
from torch.nn import functional as F


class _Moments:
    """Accumulate detached feature statistics in source-order float32."""

    def __init__(self):
        self.count = 0
        self.shape = None
        self.mean = None
        self._m2 = None
        self._counter = None

    def update(self, batch):
        batch = batch.detach().to(dtype=torch.float32)
        if self.shape is None:
            self.shape = tuple(batch.shape[1:])
            self.mean = torch.zeros(self.shape, dtype=torch.float32, device=batch.device)
            self._m2 = torch.zeros_like(self.mean)
            self._counter = torch.zeros(1, dtype=torch.long, device=batch.device)
        if tuple(batch.shape[1:]) != self.shape:
            raise ValueError("Feature shape changed during calibration")
        for sample in batch:
            old_mean = self.mean.clone()
            # Tensor division matches the source estimator arithmetic on CUDA.
            self.mean = self.mean + (sample - self.mean) / (self._counter.float() + 1)
            self._m2 = self._m2 + (sample - self.mean) * (sample - old_mean)
            self._counter += 1
            self.count += 1

    def freeze(self):
        if self.count < 2:
            raise ValueError("At least two calibration samples are required")
        return {
            "count": self.count,
            "shape": self.shape,
            "mean": self.mean.detach().clone(),
            "std": (self._m2 / (self._counter.float() - 1)).sqrt().detach().clone(),
        }


def _kl_capacity(r, logits, mean, std, min_std):
    """Return elementwise Gaussian information capacity in nats."""
    sigma = std.clamp_min(min_std)
    # Preserve the source's MC expansion: implicit broadcast changes backward reductions.
    gate = logits.sigmoid().expand(r.shape[0], *([-1] * (r.ndim - 1)))
    variance = (1 - gate).square()
    return -.5 * (1 + variance.log()
                  - (((r - mean) / sigma) * gate).square() - variance)


def _capacity_to_map(capacity, layout, size):
    """Reduce a single sample's capacity and normalize its resized map."""
    if layout == "NCHW":
        spatial = capacity.sum(dim=0)
    elif layout == "NTC":
        spatial = capacity.sum(dim=-1)
        side = isqrt(spatial.numel())
        if side * side != spatial.numel():
            raise ValueError("NTC capacity requires a square number of patch tokens")
        spatial = spatial.reshape(side, side)
    else:
        raise ValueError("Feature layout must be NCHW or NTC")
    spatial = spatial.clamp_min(0)[None, None]
    # Avoid amplifying bilinear float32 roundoff in an exactly constant map.
    if spatial.max() == spatial.min():
        return spatial.new_zeros((1, 1, *size))
    resized = F.interpolate(spatial, size=size, mode="bilinear", align_corners=False)
    return (resized - resized.min()) / (resized.max() - resized.min()).clamp_min(1e-12)


@contextmanager
def _eval_mode(model):
    """Temporarily evaluate while preserving mixed per-module mode flags."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _model_fingerprint(model):
    """Hash ordered parameters and buffers without retaining a model copy."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        metadata = repr((name, str(tensor.dtype), tuple(tensor.shape))).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()
