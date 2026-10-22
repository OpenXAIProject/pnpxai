import sys
from typing import Optional, Tuple, Any, Dict

import torch
import torch.nn.functional as F

from pnpxai.explainers.base import Explainer


DEFAULT_REPO = "kaistpnp/madical_pnpxai"
DEFAULT_CONFIG = {
    'model': 'UNet_M',
    'modality': 't2',
    'image_size': 256,
    'ddim_steps': 10,
    'vae_repo': 'farzadbz/Medical-VAE',
    'vae_filename': 'VAE-Medical-klf8.pt',
}
LATENT_SCALE = 0.18215  # latent normalization constant of the LDM VAE


def get_default_model_and_diffusion(
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    hf_repo: Optional[str] = None
) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load MAD-AD UNet, VAE, and diffusion from HuggingFace Hub.

    Returns:
        Tuple of (unet_model, vae_model, diffusion)
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    import yaml

    if hf_repo is None:
        hf_repo = DEFAULT_REPO

    config = {
        **DEFAULT_CONFIG,
        **(config if config is not None else {})
    }

    # The repository ships models.py, ldm/ and diffusion_x0/, which are imported below.
    repo_dir = snapshot_download(hf_repo)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    model_filename = config.get('model_filename', 'MAD_t2_UNet_M/checkpoints/last.pt')
    model_path = hf_hub_download(repo_id=hf_repo, filename=model_filename)

    try:
        config_path = hf_hub_download(repo_id=hf_repo, filename="MAD_t2_UNet_M/args.yml")
        with open(config_path, 'r') as f:
            hf_config = yaml.safe_load(f)
        model_name = hf_config.get('model', config['model'])
        config.update(hf_config)
    except Exception:
        model_name = config['model']

    try:
        from models import UNET_models
    except ImportError as e:
        raise ImportError(
            f"Could not import UNET_models from {hf_repo} ({repo_dir}). "
            "Please ensure models.py exists in the repository."
        ) from e

    unet_model = UNET_models[model_name](in_channels=4, out_channels=4)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    unet_model.load_state_dict(state_dict)
    unet_model.eval().to(device)

    # The VAE was pickled with ldm classes, so ldm must be importable before torch.load.
    try:
        import ldm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Could not import ldm module from {hf_repo} ({repo_dir})."
        ) from e

    vae_model_path = hf_hub_download(
        repo_id=config.get('vae_repo', DEFAULT_CONFIG['vae_repo']),
        filename=config.get('vae_filename', DEFAULT_CONFIG['vae_filename']),
    )
    vae = torch.load(vae_model_path, map_location=device)
    vae.eval().to(device)

    try:
        from diffusion_x0 import create_diffusion
    except ImportError as e:
        raise ImportError(
            f"Could not import diffusion_x0 from {hf_repo} ({repo_dir})."
        ) from e

    ddim_steps = config.get('ddim_steps', DEFAULT_CONFIG['ddim_steps'])
    diffusion = create_diffusion(
        timestep_respacing=f"ddim{ddim_steps}",
        predict_xstart=True,
        sigma_small=False,
        learn_sigma=False,
        diffusion_steps=10
    )
    return unet_model, vae, diffusion


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur of (B, H, W). Same kernel and 'reflect' boundary as
    scipy.ndimage.gaussian_filter(sigma=sigma, truncate=4.0)."""
    if sigma <= 0:
        return x
    radius = int(4.0 * sigma + 0.5)
    grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()
    out = x[:, None]
    for dim in (3, 2):
        n = out.shape[dim]
        idx = torch.arange(-radius, n + radius, device=x.device)
        idx = torch.where(idx < 0, -idx - 1, idx)
        idx = torch.where(idx >= n, 2 * n - idx - 1, idx)
        out = out.index_select(dim, idx)
        weight = kernel.view(1, 1, 1, -1) if dim == 3 else kernel.view(1, 1, -1, 1)
        out = F.conv2d(out, weight)
    return out[:, 0]


class MadXai(Explainer):
    """
    MAD-XAI: lesion localization with a masked latent diffusion model (MAD-AD) trained
    on healthy brain MRI only. No lesion labels are used at any point.

    The input is mapped to the VAE latent space and passed through the MAD-AD reverse
    process, which regenerates the latent patches the model judges abnormal and keeps
    the rest, yielding a "normal counterpart" of the input. The attribution is the
    residual between the input and that counterpart.

    The reverse process is stochastic (``eta > 0``). With ``n_samples > 1`` the explainer
    draws several counterparts and, with ``normalization='uncertainty'``, divides the
    residual by the per-pixel spread of those counterparts, so that pixels the model
    cannot reconstruct consistently (tissue boundaries, fine structure) are not flagged.

    The defaults are the configuration that reaches max-Dice 0.71 on the BraTS T2 test
    split shipped with the tutorial. ``MadXai.legacy_config()`` returns the original
    single-sample setting (max-Dice 0.65).

    Parameters:
        model: MAD-AD UNet. Loaded from HuggingFace together with ``vae`` and
            ``diffusion`` if any of the three is not given.
        vae: LDM VAE used to move between image and latent space.
        diffusion: MAD-AD diffusion object (``diffusion_x0.create_diffusion``).
        ddim_steps: Number of DDIM reverse steps per pass.
        n_samples: Number of stochastic normal counterparts drawn per input.
        eta: DDIM stochasticity. 0 is deterministic, 1 is maximal.
        mask_threshold: A latent patch is regenerated when the model's probability that it
            is normal is below this value. The original MAD-AD sampler uses 0.5.
        n_iters: Number of reverse passes; each pass starts from the previous output latent.
        normalization: ``'uncertainty'`` divides the residual by the sample standard
            deviation plus ``eps``; ``'none'`` uses the raw residual.
        sign: ``'abs'`` scores any deviation; ``'pos'``/``'neg'`` keep only pixels brighter /
            darker than the counterpart (a modality-specific assumption, off by default).
        eps: Added to the standard deviation in ``'uncertainty'`` mode.
        clip: Scores are clipped to ``[0, clip]`` and divided by ``clip``, so the output is in [0, 1].
        sigma: Standard deviation (pixels) of the final gaussian smoothing.
        seed: Seed for the sampling noise. The generator is created at the first call and
            continues across calls, so explaining a dataset batch by batch draws one noise
            stream (equivalent to ``torch.manual_seed(seed)``). ``None`` uses the global torch RNG.
        model_config: Overrides for ``DEFAULT_CONFIG`` when loading from HuggingFace.
        model_device: Device for the diffusion model and VAE.
        forward_arg_extractor, additional_forward_arg_extractor: Unused; kept for API compatibility.
    """

    SUPPORTED_MODULES = []

    def __init__(
        self,
        model=None,
        vae=None,
        diffusion=None,
        ddim_steps: int = 10,
        n_samples: int = 32,
        eta: float = 1.0,
        mask_threshold: float = 0.9,
        n_iters: int = 2,
        normalization: str = 'uncertainty',
        sign: str = 'abs',
        eps: float = 0.05,
        clip: float = 2.0,
        sigma: float = 5.0,
        seed: Optional[int] = None,
        model_config: Optional[dict] = None,
        model_device: Optional[torch.device] = None,
        forward_arg_extractor=None,
        additional_forward_arg_extractor=None,
        **kwargs,
    ):
        if model is None:
            # The base class needs a module with parameters to report a device.
            model = torch.nn.Linear(1, 1)
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)

        if normalization not in ('uncertainty', 'none'):
            raise ValueError(f"normalization must be 'uncertainty' or 'none', got {normalization!r}")
        if sign not in ('abs', 'pos', 'neg'):
            raise ValueError(f"sign must be 'abs', 'pos' or 'neg', got {sign!r}")
        if normalization == 'uncertainty' and n_samples < 2:
            raise ValueError("normalization='uncertainty' needs n_samples >= 2")

        model_device = model_device if model_device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        if vae is None or diffusion is None or isinstance(model, torch.nn.Linear):
            model, vae, diffusion = get_default_model_and_diffusion(model_config, model_device)
        self.unet_model = model
        self.vae = vae
        self.diffusion = diffusion

        self.ddim_steps = ddim_steps
        self.n_samples = n_samples
        self.eta = eta
        self.mask_threshold = mask_threshold
        self.n_iters = n_iters
        self.normalization = normalization
        self.sign = sign
        self.eps = eps
        self.clip = clip
        self.sigma = sigma
        self.seed = seed
        self.model_device = model_device
        self._generator = None

    @staticmethod
    def legacy_config() -> Dict[str, Any]:
        """Keyword arguments reproducing the pnpxai 0.1.x MadXai behaviour."""
        return dict(n_samples=1, eta=0.5, mask_threshold=0.5, n_iters=1,
                    normalization='none', sign='abs', clip=0.25, sigma=5.0)

    @property
    def device(self):
        return self.model_device

    # ------------------------------------------------------------------ sampling
    def _ddim_step(self, x, t, generator):
        """One MAD-AD DDIM step (diffusion_x0.GaussianDiffusion.ddim_sample) with the
        patch-regeneration threshold exposed as ``mask_threshold``."""
        d = self.diffusion
        out = d.p_mean_variance(self.unet_model, x, t, clip_denoised=False,
                                model_kwargs={'mask': torch.ones_like(x)})
        eps = d._predict_eps_from_xstart(x, t, out["pred_xstart"])
        shape = (-1,) + (1,) * (x.dim() - 1)
        alpha_bar = torch.as_tensor(d.alphas_cumprod, device=x.device)[t].float().view(shape)
        alpha_bar_prev = torch.as_tensor(d.alphas_cumprod_prev, device=x.device)[t].float().view(shape)
        sigma = (self.eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                 * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                     + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero = (t != 0).float().view(-1, *([1] * (x.dim() - 1)))
        sample = mean_pred + nonzero * sigma * noise
        regenerate = (out["predicted_mask"] < self.mask_threshold).to(x.dtype)
        regenerate = torch.repeat_interleave(regenerate, x.shape[1], dim=1)
        return regenerate * sample + (1 - regenerate) * x

    def _counterpart(self, latent, generator):
        """One stochastic normal counterpart of ``latent`` (n_iters reverse passes)."""
        z = latent
        for _ in range(self.n_iters):
            for i in reversed(range(self.ddim_steps)):
                t = torch.full((z.shape[0],), i, device=z.device, dtype=torch.long)
                z = self._ddim_step(z, t, generator)
        return z

    @torch.no_grad()
    def reconstruct(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Image tensor [B, 1, H, W] in [-1, 1].
        Returns:
            x0: VAE reconstruction of the input [B, C, H, W].
            counterparts: Normal counterparts [B, n_samples, C, H, W].
        """
        x = inputs.to(self.model_device)
        if self.seed is not None and self._generator is None:
            self._generator = torch.Generator(device=self.model_device).manual_seed(self.seed)
        generator = self._generator
        latent = self.vae.encode(x).mean.mul_(LATENT_SCALE)
        x0 = self.vae.decode(latent / LATENT_SCALE)
        counterparts = [self.vae.decode(self._counterpart(latent, generator) / LATENT_SCALE)
                        for _ in range(self.n_samples)]
        return x0, torch.stack(counterparts, dim=1)

    # ------------------------------------------------------------------ scoring
    def compute_anomaly_map(self, x0: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: VAE reconstruction of the input [B, C, H, W] (or [C, H, W]).
            x_reconstructed: Normal counterparts [B, S, C, H, W], or a single
                counterpart [B, C, H, W] (or [C, H, W]).
        Returns:
            Anomaly map in [0, 1], [B, H, W] (or [H, W] for unbatched input).
        """
        squeeze = x0.dim() == 3
        if squeeze:
            x0, x_reconstructed = x0.unsqueeze(0), x_reconstructed.unsqueeze(0)
        if x_reconstructed.dim() == 4:
            x_reconstructed = x_reconstructed.unsqueeze(1)
        x0 = x0.float().mean(1)                                  # [B, H, W]
        recs = x_reconstructed.float().mean(2)                   # [B, S, H, W]
        residual = x0 - recs.mean(1)
        if self.normalization == 'uncertainty':
            if recs.shape[1] < 2:
                raise ValueError("normalization='uncertainty' needs at least two counterparts")
            residual = residual / (recs.std(1) + self.eps)
        if self.sign == 'abs':
            score = residual.abs()
        elif self.sign == 'pos':
            score = residual.clamp(min=0)
        else:
            score = (-residual).clamp(min=0)
        score = score.clamp(0, self.clip) / self.clip
        score = _gaussian_blur(score, self.sigma)
        return score[0] if squeeze else score

    def attribute(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute lesion attributions for brain MRI slices.

        Args:
            inputs: Image tensor [B, 1, H, W] or [1, H, W], values in [-1, 1].
            targets: Not used (the method is label-free); kept for API compatibility.

        Returns:
            Non-negative anomaly maps in [0, 1], [B, H, W] (or [H, W]), on the input's device.
        """
        squeeze = inputs.dim() == 3
        if squeeze:
            inputs = inputs.unsqueeze(0)
        x0, counterparts = self.reconstruct(inputs)
        attrs = self.compute_anomaly_map(x0, counterparts).to(inputs.device)
        return attrs[0] if squeeze else attrs

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        """
        Tunable parameters:
            `mask_threshold` (float): patch-regeneration threshold, 0.5 to 0.95.
            `eps` (float): uncertainty floor, 0.01 to 0.2 (log scale).
            `sigma` (float): final smoothing, 1 to 9 pixels.
        """
        return {
            'mask_threshold': (float, {'low': 0.5, 'high': 0.95, 'step': 0.05}),
            'eps': (float, {'low': 0.01, 'high': 0.2, 'log': True}),
            'sigma': (float, {'low': 1.0, 'high': 9.0, 'step': 1.0}),
        }
