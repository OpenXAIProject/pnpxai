"""
MA-GIG: Manifold-Aligned Guided Integrated Gradients.

Guided Integrated Gradients (GIG) builds its integration path by greedily moving
the input features whose gradients are smallest. Because that update is
axis-aligned in pixel space, it generically points off the data manifold, and the
error accumulates over the path. MA-GIG runs the same greedy selection inside the
latent space of a pretrained VAE instead. An axis-aligned step in latent space is
mapped by the decoder Jacobian into a correlated, tangent-aligned step in pixel
space, so the decoded path stays close to the data manifold and gradients are
evaluated on plausible images.

Reference:
    Soyeon Kim, Seongwoo Lim, Kyowoon Lee, Jaesik Choi.
    Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution.
    ICML 2026. https://arxiv.org/abs/2605.02167
"""

import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core.detector.types import Convolution, Linear
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.baselines import (
    BaselineFunction,
    BaselineMethodOrFunction,
)
from pnpxai.utils import format_into_tuple


# The VAE used in the paper is the Stable Diffusion 2.1 autoencoder. This repo
# holds bit-identical weights and is the canonical publisher of them.
DEFAULT_VAE_REPO = "stabilityai/sd-vae-ft-mse"

# Normalization of the classifier's input space. The explainer needs it to move
# between the space the model consumes and the [0, 1] pixel space the VAE expects.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EPSILON = 1e-9


def slerp(t, v0: Tensor, v1: Tensor, dot_threshold: float = 0.9995) -> Tensor:
    """
    Spherical linear interpolation between two vectors.

    Interpolating along the arc rather than the chord keeps the norm of the latent
    code roughly constant, which keeps intermediate codes inside the region the
    decoder was trained on. Falls back to lerp when the vectors are nearly
    parallel (or degenerate), where the arc and the chord coincide anyway.
    """
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()

    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)

    if norm0 < 1e-9 or norm1 < 1e-9:
        return v0 * (1 - t) + v1 * t

    v0_unit = v0_flat / norm0
    v1_unit = v1_flat / norm1

    dot = torch.clamp(torch.sum(v0_unit * v1_unit), -1.0, 1.0)

    if torch.abs(dot) > dot_threshold:
        return v0 * (1 - t) + v1 * t

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0

    return s0 * v0 + s1 * v1


class VaeManifold:
    """
    Maps between the classifier's normalized input space and a VAE latent space.

    The explainer receives inputs already normalized for the classifier, while the
    VAE expects pixels in [-1, 1]. This wrapper owns that round trip so the path
    generator can stay in a single coordinate system.

    Parameters:
        vae (Module): A `diffusers` autoencoder exposing `encode`/`decode`.
        mean (Sequence[float]): Per-channel mean used to normalize classifier inputs.
        std (Sequence[float]): Per-channel std used to normalize classifier inputs.
        device (torch.device): Device holding the VAE.
    """

    def __init__(
        self,
        vae: Module,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        device: Optional[torch.device] = None,
        repo: Optional[str] = None,
    ) -> None:
        self.vae = vae.eval()
        self.device = device if device is not None else next(vae.parameters()).device
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.repo = repo

    def __repr__(self):
        return "{}(repo={})".format(self.__class__.__name__, self.repo)

    def denormalize(self, x: Tensor) -> Tensor:
        """Classifier input space -> [0, 1] pixel space."""
        return torch.stack(
            [x[:, i] * s + m for i, (m, s) in enumerate(zip(self.mean, self.std))],
            dim=1,
        )

    def normalize(self, x: Tensor) -> Tensor:
        """
        [0, 1] pixel space -> classifier input space.

        Scales channel-by-channel by a Python scalar instead of broadcasting a std
        tensor. The two agree mathematically but differ in the last bits, because
        dividing by a scalar becomes a multiply by its reciprocal, and MA-GIG
        amplifies that difference: the selection threshold sits in a dense region
        of the gradient magnitude distribution (~130 of 4096 latent dimensions
        land within 1e-7 of it), so a last-bit perturbation flips which dimensions
        move and sends the rest of the path somewhere else. Matching the reference
        implementation's arithmetic here is what keeps attributions reproducible
        against it -- do not "simplify" this to a broadcast divide.
        """
        return torch.stack(
            [(x[:, i] - m) / s for i, (m, s) in enumerate(zip(self.mean, self.std))],
            dim=1,
        )

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode a normalized classifier input to its latent posterior mean."""
        x = self.denormalize(x).to(self.device, dtype=self.vae.dtype)
        return self.vae.encode(2.0 * x - 1.0).latent_dist.mean

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode a latent back to the classifier's input space.

        Gradients flow only when `z` requires them, so the same call serves both
        the (grad-free) path construction and the (grad-carrying) latent gradient.
        """
        with torch.set_grad_enabled(z.requires_grad):
            x = self.vae.decode(z).sample
        return self.normalize((x + 1.0) / 2.0)


def load_default_vae(
    repo: str = DEFAULT_VAE_REPO,
    subfolder: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Module:
    """Load the pretrained autoencoder MA-GIG integrates through, from the Hub."""
    from diffusers import AutoencoderKL

    kwargs = {"torch_dtype": torch.float32}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    vae = AutoencoderKL.from_pretrained(repo, **kwargs)
    return vae.to(device).eval()


class MAGIG(Explainer):
    """
    MA-GIG explainer.

    Supported Modules: `Linear`, `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        vae (Optional[Module]): A `diffusers` autoencoder to integrate through. Loaded from `vae_repo` when omitted.
        vae_repo (str): HuggingFace repo id of the autoencoder, used when `vae` is not given.
        vae_subfolder (Optional[str]): Subfolder of `vae_repo` holding the autoencoder weights.
        n_steps (int): The number of steps along the integration path.
        fraction (float): Fraction of latent dimensions moved at each step, i.e. the quantile of the gradient magnitude used as the selection threshold.
        use_slerp (bool): If True, move selected latent dimensions along the arc (spherical interpolation) instead of the chord.
        exp_obj (str): Objective differentiated along the path, either `'prob'` or `'logit'`.
        baseline_fn (Optional[BaselineMethodOrFunction]): The baseline function, accepting the attribution input, and returning the baseline accordingly. Defaults to the black image, which is the baseline used in the paper. Note this is *not* the same as pnpxai's `'zeros'`, which is zero in the model's normalized space and decodes to mid-gray. The choice matters: on Oxford-IIIT Pet a blurred-input baseline scored substantially better than the paper's black one, so tune it if you care about absolute attribution quality rather than about matching published numbers.
        normalization_mean (Sequence[float]): Per-channel mean the model's inputs were normalized with.
        normalization_std (Sequence[float]): Per-channel std the model's inputs were normalized with.
        vae_device (Optional[torch.device]): Device to place the autoencoder on. Defaults to the model's device.
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).

    Notes:
        - Inputs are expected to be normalized images, i.e. exactly what `model` consumes. `normalization_mean` / `normalization_std` tell the explainer how to undo that normalization for the autoencoder.
        - Path construction is inherently sequential and decodes once per step, so runtime scales linearly in `n_steps`. Samples in a batch are processed one at a time to bound memory.
        - **Set `torch.backends.cudnn.deterministic = True` for repeatable results.** The greedy step selects latent dimensions by a low quantile of gradient magnitude, and near that threshold the distribution is dense, so a last-bit difference flips which dimensions move and the rest of the path diverges. Under PyTorch's default (non-deterministic) cuDNN setting, two identical calls return visibly different attributions; with it set, they are bit-identical.

    Reference:
        Soyeon Kim, Seongwoo Lim, Kyowoon Lee, Jaesik Choi. Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution.
    """

    SUPPORTED_MODULES = [Linear, Convolution]

    def __init__(
        self,
        model: Module,
        vae: Optional[Module] = None,
        vae_repo: str = DEFAULT_VAE_REPO,
        vae_subfolder: Optional[str] = None,
        n_steps: int = 200,
        fraction: float = 0.05,
        use_slerp: bool = True,
        exp_obj: str = "prob",
        baseline_fn: Optional[BaselineMethodOrFunction] = None,
        normalization_mean: Sequence[float] = IMAGENET_MEAN,
        normalization_std: Sequence[float] = IMAGENET_STD,
        vae_device: Optional[torch.device] = None,
        forward_arg_extractor=None,
        additional_forward_arg_extractor=None,
    ) -> None:
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        if exp_obj not in ("prob", "logit"):
            raise ValueError(f"Invalid objective function: {exp_obj}")

        self.n_steps = n_steps
        self.fraction = fraction
        self.use_slerp = use_slerp
        self.exp_obj = exp_obj
        self.baseline_fn = baseline_fn
        self.normalization_mean = tuple(normalization_mean)
        self.normalization_std = tuple(normalization_std)
        self.vae_repo = vae_repo
        self.vae_subfolder = vae_subfolder

        if self.device.type == "cuda" and not torch.backends.cudnn.deterministic:
            warnings.warn(
                "MA-GIG selects latent dimensions by a low quantile of gradient "
                "magnitude, where the distribution is dense, so last-bit "
                "differences change which dimensions move and send the path "
                "elsewhere. Under cuDNN's default non-deterministic kernel "
                "selection, repeated calls return visibly different attributions. "
                "Set torch.backends.cudnn.deterministic = True to make them "
                "reproducible.",
                stacklevel=2,
            )

        vae_device = vae_device if vae_device is not None else self.device
        if vae is None:
            vae = load_default_vae(vae_repo, vae_subfolder, vae_device)
            source = vae_repo
        else:
            source = "user-supplied"  # vae_repo says nothing about what was passed
        self.vae = VaeManifold(
            vae,
            mean=self.normalization_mean,
            std=self.normalization_std,
            device=vae_device,
            repo=source,
        )

    def _select_objective(self, outputs: Tensor, targets: Tensor) -> Tensor:
        if self.exp_obj == "prob":
            outputs = torch.softmax(outputs, dim=-1)
        return outputs[torch.arange(outputs.shape[0]), targets]

    def _baselines_for(self, inputs: Tensor) -> Tensor:
        """
        Resolve the path's starting point.

        With no `baseline_fn`, use the paper's baseline: the black image, i.e. zero
        in pixel space rather than zero in the model's normalized space.
        """
        baselines = self._get_baselines(format_into_tuple(inputs))
        if baselines is None:
            return self.vae.normalize(torch.zeros_like(inputs))
        return format_into_tuple(baselines)[0]

    def _latent_gradients(self, z: Tensor, targets: Tensor) -> Tensor:
        """Gradient of the objective w.r.t. the latent, through the decoder."""
        z = z.clone().detach().requires_grad_(True)
        outputs = self.model(self.vae.decode(z))
        obj = self._select_objective(outputs, targets)
        return torch.autograd.grad(obj.sum(), z)[0].detach()

    def _slerp_update(
        self, z: Tensor, z_target: Tensor, gamma: Tensor, mask: Tensor
    ) -> Tensor:
        z_new = z.clone()
        if mask.sum() == 0:
            return z_new
        z_new[mask] = slerp(gamma, z[mask], z_target[mask])
        return z_new

    def generate_path(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Build the manifold-aligned path for a single sample.

        Exposed because the path is what distinguishes MA-GIG: decoding it shows
        the intermediate images the attribution actually integrates over, which is
        the most direct way to see the method working.

        Walks from the baseline latent toward the input latent, at each step moving
        only the `fraction` of latent dimensions with the smallest gradient
        magnitude, and decoding the result. Endpoints are anchored to the raw
        baseline and input images so the path terminates exactly at them
        regardless of the autoencoder's reconstruction error.

        Args:
            inputs (Tensor): A single normalized image, shaped [1, C, H, W].
            targets (Tensor): The target label, shaped [1].

        Returns:
            Tensor: The path, shaped [n_steps, C, H, W].
        """
        baselines = self._baselines_for(inputs)

        z_input = self.vae.encode(inputs).squeeze(0)
        z_baseline = self.vae.encode(baselines).squeeze(0)

        x_input_raw = inputs.squeeze(0)
        x_baseline_raw = baselines.squeeze(0)

        z = z_baseline.clone()
        z_max = z_input
        l1_total = torch.abs(z_input - z_baseline).sum()

        path = []
        for step in range(self.n_steps):
            if step == 0:
                path.append(x_baseline_raw.clone())
            elif step == self.n_steps - 1:
                path.append(x_input_raw.clone())
                break
            else:
                path.append(self.vae.decode(z.unsqueeze(0)).squeeze(0).clone())

            grad = self._latent_gradients(z[None], targets)[0].clone()

            # Distance still to cover once this step is done.
            l1_target = l1_total * (1 - (step + 1) / self.n_steps)

            gamma = float("inf")
            while gamma > 1.0:
                l1_current = torch.abs(z - z_input).sum()
                if torch.isclose(l1_target, l1_current, rtol=EPSILON, atol=EPSILON):
                    break

                # Dimensions already at the target cannot absorb more movement;
                # push them out of the selection.
                at_max = torch.abs(z - z_max) < EPSILON
                grad = torch.where(
                    at_max, torch.tensor(float("inf"), device=z.device), grad
                )

                threshold = torch.quantile(
                    grad.abs().reshape(-1), self.fraction, interpolation="lower"
                )
                selected = (torch.abs(grad) <= threshold) & (grad != float("inf"))

                # How far the selected dimensions could move in total.
                l1_selected = (torch.abs(z - z_max) * selected).sum()
                gamma = (
                    (l1_current - l1_target) / l1_selected
                    if l1_selected > 0
                    else float("inf")
                )

                if gamma > 1.0:
                    # Not enough budget in this selection; saturate it and
                    # re-select on the next pass.
                    z = torch.where(selected, z_max, z)
                elif self.use_slerp:
                    z = self._slerp_update(z, z_max, gamma, selected)
                else:
                    z = torch.where(selected, z + (z_max - z) * gamma, z)

        return torch.stack(path, dim=0)

    def _accumulate(self, path: Tensor, targets: Tensor) -> Tensor:
        """Riemann sum of gradient x displacement along the decoded path."""
        grads = torch.zeros_like(path)
        for i in range(path.shape[0]):
            point = path[i : i + 1].clone().requires_grad_(True)
            obj = self._select_objective(self.model(point), targets)
            grads[i] = torch.autograd.grad(obj.sum(), point)[0].detach().squeeze(0)

        deltas = path[1:] - path[:-1]
        return (deltas * grads[:-1]).sum(dim=0)

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor,
    ) -> Tensor:
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            torch.Tensor: The result of the explanation.
        """
        forward_args, _ = self._extract_forward_args(inputs)
        forward_args = format_into_tuple(forward_args)[0]

        # Accept a bare class index as well as a per-sample tensor of them.
        if not isinstance(targets, Tensor):
            targets = torch.as_tensor(targets, device=forward_args.device)
        if targets.dim() == 0:
            targets = targets.reshape(1).expand(forward_args.shape[0])

        attrs = []
        for sample, target in zip(forward_args, targets):
            sample = sample[None]
            target = target[None]
            path = self.generate_path(sample, target)
            attrs.append(self._accumulate(path, target))
        return torch.stack(attrs, dim=0)

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        """
        Provides Tunable parameters for the optimizer

        Tunable parameters:
            `n_steps` (int): Value can be selected in the range of `range(50, 300, 50)`

            `fraction` (float): Value can be selected in the range of `range(0.05, 0.5, 0.05)`

            `use_slerp` (bool): Value can be selected among `[True, False]`

            `baseline_fn` (callable): BaselineFunction selects suitable values in accordance with the modality
        """
        return {
            "n_steps": (int, {"low": 50, "high": 300, "step": 50}),
            "fraction": (float, {"low": 0.05, "high": 0.5, "step": 0.05}),
            "use_slerp": (list, {"choices": [True, False]}),
            "baseline_fn": (BaselineFunction, {}),
        }
