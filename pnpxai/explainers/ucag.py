"""UCAG with raw GradCAM partial maps (Hong et al., AAAI 2023).

Reference: https://doi.org/10.1609/aaai.v37i7.25954
Native implementation: https://github.com/KU-HJH/UCAG
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from captum.attr import LayerGradCam

from pnpxai.core.detector.types import Convolution
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils import find_cam_target_layer


class UCAG(Explainer):
    """Confidence-weighted crop aggregation for one image classifier input.

    Args:
        model (Module): Classifier returning a (batch, classes) logits tensor.
        layer (Optional[Union[str, Module]]): Target module or dotted name;
            None enables CAM layer discovery.
        crop_size (int): Crop side at the reference input size of 224.
        num_units (int): Grid side; produces num_units ** 2 crops per image.
        alpha (float): Crop enlargement factor.
        limit_batch_size (int): Maximum crops per forward/backward call.
        score (bool): Weight partial maps by exp(target softmax probability).
        **kwargs (Any): Forward-argument extractors accepted by Explainer.

    Attributions are detached non-negative maps (B, 1, H, W) on the input
    device/dtype. Constant maps return zeros. No auxiliary model is loaded.
    """

    SUPPORTED_MODULES = [Convolution]

    def __init__(self, model, layer=None, crop_size=124, num_units=6,
                 alpha=2.6, limit_batch_size=16, score=True, **kwargs):
        super().__init__(model, **kwargs)
        self.layer = layer
        self.crop_size = crop_size
        self.num_units = num_units
        self.alpha = alpha
        self.limit_batch_size = limit_batch_size
        self.score = score

    def set_target_layer(self, layer):
        return self.set_kwargs(layer=layer)

    def get_tunables(self):
        return {
            "alpha": (float, {"low": 1.0, "high": 3.0}),
            "num_units": (int, {"low": 2, "high": 8}),
        }

    def attribute(self, inputs, targets):
        images, additional = self._extract_forward_args(inputs)
        if isinstance(images, tuple) and len(images) == 1:
            images = images[0]
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or not images.is_floating_point()):
            raise ValueError("UCAG requires one floating-point NCHW image tensor")
        if additional is not None and not (isinstance(additional, tuple) and not additional):
            raise ValueError("UCAG does not support additional forward arguments")
        if images.device != self.device:
            raise ValueError("Images and classifier must be on the same device")
        if images.shape[0] == 0 or min(images.shape[-2:]) < 3:
            raise ValueError("UCAG requires a nonempty batch and spatial dimensions >= 3")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("UCAG requires autograd; disable torch.inference_mode")
        for name, minimum in (("crop_size", 1), ("num_units", 2), ("limit_batch_size", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.alpha, (float, int)) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        labels = torch.as_tensor(targets, device=images.device)
        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("Targets must contain integer class indices")
        if labels.ndim == 0:
            labels = labels.expand(len(images))
        if labels.shape != (len(images),) or (labels < 0).any():
            raise ValueError("Targets must be a scalar or a nonnegative vector of batch length")
        labels = labels.long()
        layer = self.layer
        if isinstance(layer, str):
            try:
                layer = self.model.get_submodule(layer)
            except AttributeError as error:
                raise ValueError(f"Unknown target layer: {layer}") from error
        if layer is None:
            layer = find_cam_target_layer(self.model)
        if not isinstance(layer, nn.Module) or not any(layer is m for m in self.model.modules()):
            raise ValueError("Target layer must belong to the classifier")
        with torch.enable_grad():
            return torch.cat([self._attribute_image(image[None], int(label), layer)
                              for image, label in zip(images, labels)]).detach()

    def _attribute_image(self, image, target, layer):
        height, width = image.shape[-2:]
        ch, cw = int(self.crop_size * height / 224), int(self.crop_size * width / 224)
        uh, uw = int(ch * self.alpha), int(cw * self.alpha)
        if min(ch, cw, uh, uw) < 1 or ch > height or cw > width:
            raise ValueError("Crop and enlarged crop dimensions must be positive and fit the input")
        n = self.num_units
        # Preserve the original implementation's integer spacing, including +2.
        sh, sw = (height - ch + 2) // (n - 1), (width - cw + 2) // (n - 1)
        crops = []
        for i in range(n):
            for j in range(n):
                crop = image[:, :, i * sh:i * sh + ch, j * sw:j * sw + cw]
                if min(crop.shape[-2:]) == 0:
                    raise ValueError("Grid geometry produced an empty crop")
                crops.append(F.interpolate(crop, (uh, uw), mode="bicubic", align_corners=False))
        crops = torch.cat(crops).detach().requires_grad_(True)
        maps, probabilities = [], []
        for chunk in crops.split(self.limit_batch_size):
            logits = None

            def forward(x):
                nonlocal logits
                result = self.model(x)
                if not isinstance(result, torch.Tensor) or result.ndim != 2:
                    raise ValueError("Classifier must return a (batch, classes) logits tensor")
                if target >= result.shape[1]:
                    raise ValueError("Target class is outside the classifier output")
                logits = result.detach()
                return result

            raw = LayerGradCam(forward, layer).attribute(
                chunk, target=target, relu_attributions=False,
            )
            maps.append(raw.detach())
            probabilities.append(logits.softmax(1)[:, target])
        maps = F.interpolate(torch.cat(maps), (ch, cw), mode="bilinear", align_corners=False)
        weights = torch.cat(probabilities).exp() if self.score else image.new_ones(n * n)
        weights = weights / weights.max()
        canvas = image.new_zeros(1, 1, ch + (n - 1) * sh, cw + (n - 1) * sw)
        divisor = torch.ones_like(canvas)
        for i in range(n):
            for j in range(n):
                region = (slice(None), slice(None), slice(i * sh, i * sh + ch),
                          slice(j * sw, j * sw + cw))
                canvas[region] += maps[i * n + j] * weights[i * n + j]
                divisor[region] += 1
        result = F.interpolate((canvas / divisor).relu(), (height, width),
                               mode="bilinear", align_corners=False)
        result = result - result.amin()
        scale = result.amax()
        result = result / scale if scale > 0 else torch.zeros_like(result)
        # Separable Gaussian5/sigma3, reflect padding, matching native UCAG.
        coordinates = torch.arange(-2, 3, device=image.device, dtype=image.dtype)
        kernel = torch.exp(-coordinates.square() / 18)
        kernel = kernel / kernel.sum()
        result = F.conv2d(F.pad(result, (2, 2, 0, 0), mode="reflect"), kernel.view(1, 1, 1, 5))
        return F.conv2d(F.pad(result, (0, 0, 2, 2), mode="reflect"), kernel.view(1, 1, 5, 1))
