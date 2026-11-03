"""Per-image Gaussian CoIBA with explicit reference-feature calibration."""

import copy
import math
import os
import tempfile
from inspect import signature
from numbers import Integral, Real
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pnpxai.core.detector.types import Convolution, Linear, Attention
from pnpxai.explainers.base import Explainer
from pnpxai.explainers._coiba_utils import (
    _Moments, _capacity_to_map, _eval_mode, _kl_capacity, _model_fingerprint,
)


class CoIBA(Explainer):
    """Optimize a shared information bottleneck over selected image features.

    Supported Modules: `Convolution`, `Linear`, `Attention`

    Call :meth:`estimate` on a standard ``(images, labels)`` loader before
    attribution. ``NCHW`` features use spatial masks; ``NTC`` features use a
    square patch grid and optionally preserve leading prefix tokens. Noise is
    seeded separately for every image; reproducibility across devices and
    PyTorch versions is not guaranteed.

    AutoExplanation does not recommend CoIBA because it needs layers and
    statistics first; add a prepared instance with
    ``experiment.manager.add_explainer``.

    Parameters:
        model (Module): Image classifier returning ``(batch, classes)`` logits.
        layer (Union[str, Module, Sequence[Union[str, Module]]]): Module,
            module name, or a sequence of them sharing one mask.
        feature_layout (str): ``"NCHW"`` for convolutional or ``"NTC"`` for
            token features.
        skip_prefix_tokens (int): Leading ``"NTC"`` tokens, such as a class
            token, left unperturbed and out of the map. The official
            implementation instead perturbs them and drops them only from
            the map.
        beta (float): Weight of the information loss against the
            classification loss.
        optimization_steps (int): Adam steps per image.
        lr (float): Adam learning rate.
        batch_size (int): Noise samples per optimization step.
        min_std (float): Lower bound on feature standard deviations.
        seed (int): Noise seed, reset for every image.
        forward_arg_extractor (Optional[Callable]): Extracts the image tensor
            from a data batch.
        additional_forward_arg_extractor (Optional[Callable]): Must extract
            nothing; additional forward arguments are unsupported.

    Reference:
        Jung-Ho Hong, Ho-Joong Kim, Kyu-Sung Jeon, Seong-Whan Lee. Comprehensive Information Bottleneck for Unveiling Universal Attribution to Interpret Vision Transformers. CVPR 2025.
        Karl Schulz, Leon Sixt, Federico Tombari, Tim Landgraf. Restricting the Flow: Information Bottlenecks for Attribution. ICLR 2020.
    """

    SUPPORTED_MODULES = [Convolution, Linear, Attention]

    def __init__(self, model, layer=None, feature_layout="NCHW", skip_prefix_tokens=0,
                 beta=10, optimization_steps=10, lr=1, batch_size=10, min_std=.01,
                 seed=0, forward_arg_extractor=None, additional_forward_arg_extractor=None):
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.layer = tuple(layer) if isinstance(layer, (tuple, list)) else layer
        self.feature_layout = feature_layout
        self.skip_prefix_tokens = skip_prefix_tokens
        self.beta = beta
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.min_std = min_std
        self.seed = seed
        self._statistics = None
        self._input_shape = None
        self._prepared_fingerprint = None
        self._prepared_configuration = None
        self._validate_configuration()

    def get_tunables(self):
        """Tune the bottleneck tradeoff while retaining prepared statistics."""
        return {"beta": (float, {"low": 0.01, "high": 100.0, "log": True})}

    def copy(self) -> "CoIBA":
        """Shallow-copy configuration, sharing statistics that calls never mutate."""
        clone = copy.copy(self)
        if isinstance(clone.layer, (tuple, list)):
            clone.layer = tuple(clone.layer)
        return clone

    def set_kwargs(self, **kwargs) -> "CoIBA":
        """Return a validated copy; feature/model changes require preparation."""
        unknown = kwargs.keys() - signature(type(self).__init__).parameters.keys()
        if unknown or "self" in kwargs:
            raise TypeError(f"Unknown CoIBA constructor argument(s): {sorted(unknown or {'self'})}")
        clone = self.copy()
        for name, value in kwargs.items():
            setattr(clone, name, tuple(value) if name == "layer" and isinstance(value, (tuple, list)) else value)
        clone._validate_configuration()
        names = tuple(name for name, _ in clone._resolve_layers()) if clone.layer is not None else ()
        configuration = (names, clone.feature_layout, clone.skip_prefix_tokens)
        if clone.model is not self.model or configuration != self._prepared_configuration:
            clone._statistics, clone._input_shape = None, None
            clone._prepared_fingerprint, clone._prepared_configuration = None, None
        return clone

    def _validate_configuration(self):
        if not isinstance(self.model, nn.Module):
            raise ValueError("model must be a torch module")
        if self.feature_layout not in ("NCHW", "NTC"):
            raise ValueError("feature_layout must be NCHW or NTC")
        for name, minimum in (("skip_prefix_tokens", 0), ("optimization_steps", 1),
                              ("batch_size", 1), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.feature_layout == "NCHW" and self.skip_prefix_tokens:
            raise ValueError("skip_prefix_tokens is only supported for NTC features")
        for name in ("beta", "lr", "min_std"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0
                    or (name != "beta" and value == 0)):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'beta' else 'positive'}")

    def _resolve_layers(self):
        """Resolve unique owned modules in named_modules traversal order."""
        selected = self.layer
        if isinstance(selected, (str, nn.Module)):
            selected = (selected,)
        if not isinstance(selected, (tuple, list)) or not selected:
            raise ValueError("Select at least one layer before estimate or attribute")
        named = tuple(self.model.named_modules())
        by_name = dict(named)
        names = []
        for layer in selected:
            if isinstance(layer, str) and layer in by_name:
                name = layer
            elif isinstance(layer, nn.Module):
                name = next((name for name, module in named if module is layer), None)
            else:
                name = None
            if name is None:
                raise ValueError(f"Selected layer/module {layer!r} does not belong to model")
            if name in names:
                raise ValueError(f"Duplicate selected layer: {name}")
            names.append(name)
        return tuple((name, module) for name, module in named if name in names)

    def _require_prepared(self):
        self._validate_configuration()
        if not self._statistics:
            raise RuntimeError("CoIBA needs feature statistics: call estimate or load_statistics first")
        names = tuple(name for name, _ in self._resolve_layers())
        configuration = (names, self.feature_layout, self.skip_prefix_tokens)
        if configuration != self._prepared_configuration:
            raise RuntimeError("Selected layer/layout configuration changed; estimate statistics again")
        if _model_fingerprint(self.model) != self._prepared_fingerprint:
            raise RuntimeError("Model fingerprint changed; estimate statistics again")

    def _validate_image(self, image, shape=None):
        if not isinstance(image, torch.Tensor):
            raise TypeError("CoIBA requires a single image tensor")
        if image.dtype != torch.float32 or image.ndim != 4 or any(d <= 0 for d in image.shape):
            raise ValueError("Images must be a nonempty float32 NCHW tensor")
        if image.device != self.device:
            raise ValueError("Image device must match classifier device")
        if shape is not None and tuple(image.shape[1:]) != shape:
            raise ValueError("Image shape must match calibration input shape")
        if not image.isfinite().all():
            raise ValueError("Images must contain finite values")

    def _validate_feature(self, output, batch_size, shape=None):
        if not isinstance(output, torch.Tensor):
            raise ValueError("Selected layer output must be a tensor")
        ndim = 4 if self.feature_layout == "NCHW" else 3
        if (output.ndim != ndim or output.shape[0] != batch_size
                or output.dtype != torch.float32 or output.device != self.device
                or any(d <= 0 for d in output.shape)):
            raise ValueError("Selected layer output must match float32 feature layout/device/batch")
        if shape is not None and tuple(output.shape[1:]) != shape:
            raise ValueError("Selected layer feature shape changed")
        if not output.isfinite().all():
            raise ValueError("Selected layer output must contain finite values")
        if self.feature_layout == "NTC":
            patches = output.shape[1] - self.skip_prefix_tokens
            if patches <= 0 or math.isqrt(patches)**2 != patches:
                raise ValueError("NTC features require a square number of non-prefix patch tokens")

    def estimate(self, loader, n_samples=1000):
        """Collect sample mean/std from at most ``n_samples`` loader images.

        The loader supplies ``(images, labels)``; labels are ignored. Images
        are moved to the classifier device. A failed estimate leaves previously
        prepared statistics intact.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError("CoIBA calibration cannot run in inference_mode")
        self._validate_configuration()
        layers = self._resolve_layers()
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 2:
            raise ValueError("n_samples must be an integer of at least two samples")
        moments = {name: _Moments() for name, _ in layers}
        handles, seen, input_shape = [], 0, None
        index, current_batch = 0, 0

        def make_hook(name):
            def hook(module, args, output):
                nonlocal index
                if index >= len(layers) or layers[index][0] != name:
                    raise ValueError("Selected layers must execute exactly once in traversal order")
                self._validate_feature(output, current_batch)
                moments[name].update(output.detach())
                index += 1
            return hook

        try:
            for name, module in layers:
                handles.append(module.register_forward_hook(make_hook(name)))
            # Isolate CPU and the classifier's CUDA RNG without touching other GPUs.
            rng_devices = [self.device.index] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices), torch.no_grad(), _eval_mode(self.model):
                for batch in loader:
                    image = batch[0] if isinstance(batch, (tuple, list)) else batch
                    if not isinstance(image, torch.Tensor):
                        raise TypeError("Calibration loader must provide image tensors")
                    image = image[:n_samples-seen].to(self.device)
                    self._validate_image(image, input_shape)
                    input_shape = tuple(image.shape[1:])
                    index, current_batch = 0, image.shape[0]
                    self.model(image)
                    if index != len(layers):
                        raise ValueError("Not all selected layers executed during calibration")
                    seen += current_batch
                    if seen == n_samples:
                        break
                if seen != n_samples:
                    raise ValueError(f"Insufficient calibration samples: expected {n_samples}, got {seen}")
                statistics = {name: value.freeze() for name, value in moments.items()}
                if any(not s[key].isfinite().all() for s in statistics.values() for key in ("mean", "std")):
                    raise ValueError("Calibration statistics must be finite")
                fingerprint = _model_fingerprint(self.model)
        finally:
            for handle in handles:
                handle.remove()
        self._statistics = statistics
        self._input_shape = input_shape
        self._prepared_fingerprint = fingerprint
        self._prepared_configuration = (tuple(name for name, _ in layers),
                                        self.feature_layout, self.skip_prefix_tokens)
        return self

    @staticmethod
    def _validate_preprocessing_id(preprocessing_id):
        if not isinstance(preprocessing_id, str) or not preprocessing_id.strip():
            raise ValueError("preprocessing_id must be a nonempty string")

    @staticmethod
    def _cache_shape(shape, rank, label):
        if (not isinstance(shape, tuple) or len(shape) != rank
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError(f"Cache {label} must contain {rank} positive integer dimensions")
        return shape

    def _validate_cache(self, payload, preprocessing_id):
        """Validate the complete primitive/tensor payload before state publication."""
        required = {"format_version", "model_fingerprint", "layer_names", "feature_layout",
                    "skip_prefix_tokens", "input_shape", "dtype", "n_samples",
                    "preprocessing_id", "layers"}
        if not isinstance(payload, dict) or payload.keys() != required:
            raise ValueError("Cache schema must contain exactly the required keys")
        if type(payload["format_version"]) is not int or payload["format_version"] != 1:
            raise ValueError("Unsupported cache format_version")
        self._validate_preprocessing_id(payload["preprocessing_id"])
        if payload["preprocessing_id"] != preprocessing_id:
            raise ValueError("Cache preprocessing_id does not match")
        names = tuple(name for name, _ in self._resolve_layers())
        if (not isinstance(payload["layer_names"], tuple)
                or any(not isinstance(name, str) for name in payload["layer_names"])
                or payload["layer_names"] != names):
            raise ValueError("Cache layer_names must match selected names and order")
        if not isinstance(payload["feature_layout"], str) or payload["feature_layout"] != self.feature_layout:
            raise ValueError("Cache feature_layout does not match")
        if (type(payload["skip_prefix_tokens"]) is not int
                or payload["skip_prefix_tokens"] != self.skip_prefix_tokens):
            raise ValueError("Cache skip_prefix_tokens does not match")
        shape = self._cache_shape(payload["input_shape"], 3, "input_shape")
        if not isinstance(payload["dtype"], str) or payload["dtype"] != "torch.float32":
            raise ValueError("Cache dtype must be torch.float32")
        count = payload["n_samples"]
        if type(count) is not int or count < 2:
            raise ValueError("Cache n_samples count must be an integer >= 2")
        fingerprint = _model_fingerprint(self.model)
        if (not isinstance(payload["model_fingerprint"], str)
                or payload["model_fingerprint"] != fingerprint):
            raise ValueError("Cache model fingerprint does not match current model")
        layers = payload["layers"]
        if not isinstance(layers, dict) or tuple(layers) != names:
            raise ValueError("Cache layers must match selected names and order")
        for name, state in layers.items():
            if not isinstance(state, dict) or state.keys() != {"count", "shape", "mean", "std"}:
                raise ValueError(f"Cache layer {name} schema must contain the required keys")
            if type(state["count"]) is not int or state["count"] != count:
                raise ValueError(f"Cache layer {name} count must match n_samples")
            feature_shape = self._cache_shape(state["shape"], 3 if self.feature_layout == "NCHW" else 2,
                                              f"layer {name} feature shape")
            if self.feature_layout == "NTC":
                patches = feature_shape[0] - self.skip_prefix_tokens
                if patches <= 0 or math.isqrt(patches)**2 != patches:
                    raise ValueError(f"Cache layer {name} feature shape requires square patch tokens")
            for key in ("mean", "std"):
                tensor = state[key]
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Cache layer {name} {key} must be a tensor")
                if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
                    raise ValueError(f"Cache layer {name} {key} dtype/layout must be dense float32")
                if tuple(tensor.shape) != feature_shape:
                    raise ValueError(f"Cache layer {name} {key} shape does not match feature shape")
                if not tensor.isfinite().all():
                    raise ValueError(f"Cache layer {name} {key} must be finite")
                if key == "std" and (tensor < 0).any():
                    raise ValueError(f"Cache layer {name} std must be nonnegative")
        configuration = (names, self.feature_layout, self.skip_prefix_tokens)
        return layers, shape, fingerprint, configuration

    def save_statistics(self, path, *, preprocessing_id: str) -> None:
        """Atomically write a portable tensor/primitive Gaussian statistics cache."""
        self._validate_preprocessing_id(preprocessing_id)
        self._require_prepared()
        payload = {
            "format_version": 1,
            "model_fingerprint": self._prepared_fingerprint,
            "layer_names": tuple(name for name, _ in self._resolve_layers()),
            "feature_layout": self.feature_layout,
            "skip_prefix_tokens": int(self.skip_prefix_tokens),
            "input_shape": tuple(self._input_shape),
            "dtype": "torch.float32",
            "n_samples": next(iter(self._statistics.values()))["count"],
            "preprocessing_id": preprocessing_id,
            "layers": {name: {"count": state["count"], "shape": tuple(state["shape"]),
                              "mean": state["mean"].detach().cpu().clone(),
                              "std": state["std"].detach().cpu().clone()}
                       for name, state in self._statistics.items()},
        }
        self._validate_cache(payload, preprocessing_id)
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = stream.name
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_statistics(self, path, *, preprocessing_id: str) -> "CoIBA":
        """Validate a weights-only cache, preserving prior preparation on failure."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("CoIBA statistics cannot be loaded in inference_mode")
        self._validate_preprocessing_id(preprocessing_id)
        self._validate_configuration()
        payload = torch.load(path, map_location=self.device, weights_only=True)
        layers, shape, fingerprint, configuration = self._validate_cache(payload, preprocessing_id)
        # Own the loaded dictionary and detached tensors; attribution only reads them.
        statistics = {name: {"count": state["count"], "shape": state["shape"],
                             "mean": state["mean"].detach().clone(),
                             "std": state["std"].detach().clone()}
                      for name, state in layers.items()}
        (self._statistics, self._input_shape, self._prepared_fingerprint,
         self._prepared_configuration) = statistics, shape, fingerprint, configuration
        return self

    def attribute(self, inputs, targets):
        """Return detached normalized information maps shaped ``(B,1,H,W)``."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("CoIBA requires autograd and cannot run in inference_mode")
        self._require_prepared()
        image, additional = self._extract_forward_args(inputs)
        if additional is not None and not (isinstance(additional, (tuple, list)) and len(additional) == 0):
            raise ValueError("CoIBA does not support additional forward arguments")
        if isinstance(image, tuple):
            if len(image) != 1:
                raise ValueError("CoIBA requires one extracted image input")
            image = image[0]
        self._validate_image(image, self._input_shape)
        if isinstance(targets, Integral) and not isinstance(targets, bool):
            targets = torch.full((image.shape[0],), int(targets), dtype=torch.long, device=image.device)
        elif isinstance(targets, torch.Tensor) and targets.dtype in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if targets.ndim == 0:
                targets = targets.expand(image.shape[0])
            if targets.ndim != 1 or targets.shape[0] != image.shape[0]:
                raise ValueError("Targets must be scalar or an integer vector of input batch length")
        else:
            raise ValueError("Targets must be scalar integers or integer tensors")
        if (targets < 0).any():
            raise ValueError("Targets must be valid nonnegative class indices")
        layers = self._resolve_layers()
        with torch.enable_grad():
            return torch.cat([self._attribute_image(image[i:i+1].detach(), int(target), layers)[0]
                              for i, target in enumerate(targets)]).detach()

    def _attribute_image(self, image, target, layers):
        """Optimize one image; raw capacity omits only the Monte Carlo batch."""
        first_name = layers[0][0]
        shape = self._statistics[first_name]["shape"]
        mask_shape = (1, *shape[1:]) if self.feature_layout == "NCHW" else (shape[0]-self.skip_prefix_tokens, 1)
        alpha = torch.full(mask_shape, 5., dtype=image.dtype, device=image.device, requires_grad=True)
        optimizer = torch.optim.Adam([alpha], lr=self.lr)
        gradient_inputs = (alpha, *(parameter for parameter in self.model.parameters() if parameter.requires_grad))
        generator = torch.Generator(device=image.device).manual_seed(self.seed)
        handles, activations = [], {}
        index, capacity = 0, None

        def reset_iteration_buffers():
            nonlocal index, capacity
            index, capacity = 0, None
            activations.clear()

        def make_hook(name):
            def hook(module, args, output):
                nonlocal index, capacity
                if index >= len(layers) or layers[index][0] != name:
                    raise ValueError("Selected layers must execute exactly once in traversal order")
                state = self._statistics[name]
                self._validate_feature(output, self.batch_size, state["shape"])
                prefix = self.skip_prefix_tokens if self.feature_layout == "NTC" else 0
                feature = output[:, prefix:] if prefix else output
                mean = state["mean"][prefix:] if prefix else state["mean"]
                std = state["std"][prefix:] if prefix else state["std"]
                # Clone protects first-layer capacity from later in-place modules.
                activations[name] = feature.clone()
                index += 1
                if index == len(layers):
                    first = self._statistics[first_name]
                    first_mean = first["mean"][prefix:] if prefix else first["mean"]
                    first_std = first["std"][prefix:] if prefix else first["std"]
                    capacity = _kl_capacity(activations[first_name], alpha,
                                            first_mean, first_std, self.min_std)
                gate = alpha.sigmoid()
                if self.feature_layout == "NCHW" and gate.shape[-2:] != feature.shape[-2:]:
                    gate = F.interpolate(gate[None], feature.shape[-2:], mode="bilinear", align_corners=False)[0]
                if self.feature_layout == "NTC" and gate.shape[0] != feature.shape[1]:
                    raise ValueError("Selected NTC layers must have compatible patch-token counts")
                gate = gate.expand(feature.shape[0], *([-1] * (feature.ndim - 1)))
                # Source x.data.new(x.size()) assigns the RNG stream in contiguous order.
                eps = torch.empty_like(feature, memory_format=torch.contiguous_format).normal_(generator=generator)
                noise = std.clamp_min(self.min_std) * eps + mean
                restricted = gate * feature + (1-gate) * noise
                return torch.cat((output[:, :prefix], restricted), 1) if prefix else restricted
            return hook

        try:
            for name, module in layers:
                handles.append(module.register_forward_hook(make_hook(name)))
            rng_devices = [self.device.index] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=rng_devices), torch.enable_grad(), _eval_mode(self.model):
                for _ in range(self.optimization_steps):
                    optimizer.zero_grad(set_to_none=True)
                    reset_iteration_buffers()
                    logits = self.model(image.expand(self.batch_size, -1, -1, -1))
                    if index != len(layers) or capacity is None:
                        raise ValueError("Not all selected layers executed during attribution")
                    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                            or logits.shape[0] != self.batch_size or logits.shape[1] <= target):
                        raise ValueError("Classifier logits must be (internal_batch, classes) with valid target bounds")
                    if not logits.isfinite().all() or not capacity.isfinite().all():
                        raise ValueError("Nonfinite CoIBA logits/capacity; check model, features, and min_std")
                    target_batch = torch.full((self.batch_size,), target, device=image.device, dtype=torch.long)
                    information_loss = capacity.mean(dim=0).mean()
                    loss = F.cross_entropy(logits, target_batch) + self.beta * information_loss
                    if not loss.isfinite():
                        raise ValueError("Nonfinite CoIBA loss; check beta, min_std, and model outputs")
                    # Preserve source backward branches without accumulating classifier .grad.
                    # Index immediately so transient classifier gradients are freed before Adam.
                    alpha.grad = torch.autograd.grad(loss, gradient_inputs, allow_unused=True)[0]
                    if not alpha.grad.isfinite().all():
                        raise ValueError("Nonfinite CoIBA mask gradient; check optimization configuration")
                    optimizer.step()
                raw = capacity.detach().mean(0)
                return _capacity_to_map(raw, self.feature_layout, image.shape[-2:]), raw
        finally:
            for handle in handles:
                handle.remove()
