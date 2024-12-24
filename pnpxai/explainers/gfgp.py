import os
import sys
from typing import Optional, Sequence, Tuple, Any

import numpy as np
from PIL import Image
from scipy.special import softmax

import torch
import torchvision.transforms as T

from .base import Explainer


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


DEFAULT_REPO = "devilops/blended-diffusion-custom"
DEFAULT_CONFIG = {
    'image_size': 256,  # 256
    'num_channels': 256,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'num_heads': 4,
    'num_heads_upsample': -1,
    'num_head_channels': 64,
    'attention_resolutions': '32, 16, 8',
    'channel_mult': '',
    'dropout': 0.0,
    'class_cond': False,
    'use_checkpoint': False,
    'use_scale_shift_norm': True,
    'use_fp16': True,  # True,
    'use_new_attention_order': False,
    'learn_sigma': True,
    'diffusion_steps': 1000,
    'noise_schedule': 'linear',
    'timestep_respacing': "1000",
    'use_kl': False,
    'predict_xstart': False,
    'rescale_timesteps': True,
    'rescale_learned_sigmas': False
}


def get_default_model_and_diffusion(
    config: Optional[dict] = None, device: Optional[torch.device] = None, hf_repo: Optional[str] = None
) -> Tuple[torch.nn.Module, Any]:
    from huggingface_hub import snapshot_download

    if hf_repo is None:
        hf_repo = DEFAULT_REPO

    repo_dir = snapshot_download(hf_repo)
    # appending a path
    sys.path.append(repo_dir)
    from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion

    config = {
        **DEFAULT_CONFIG,
        **(config if config is not None else {})
    }
    model, diffusion = create_model_and_diffusion(**config)

    diffusion_ckpt_path = os.path.join(
        repo_dir, "diffusion_ckpts/256x256_diffusion_uncond.pt"
    )
    model.load_state_dict(torch.load(diffusion_ckpt_path, map_location=device))
    model = model.to(device)

    model.requires_grad_(False).eval()
    if config["use_fp16"]:
        model.convert_to_fp16()

    return model, diffusion


class Gfgp(Explainer):
    def __init__(
        self,
        model,  # classification model to be explained
        transforms: Optional[callable] = None,
        timesteps: Optional[Sequence[int]] = None,
        n_perturbations=100,
        model_config=None,
        model_device=None,
        forward_arg_extractor=None,
        additional_forward_arg_extractor=None,
    ):
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)

        self.transforms = transforms

        model_device = model_device if model_device is not None else self.device
        self.diffusion_model, self.diffusion = get_default_model_and_diffusion(
            model_config, model_device
        )

        self.timesteps = timesteps if timesteps is not None else [300]
        self.n_perturbations = n_perturbations

    def tweedie_perturb(self, x, idx: int = 300, n_perturbations: int = 100):
        """
        Generate perturbations using diffusion
        The perturbations are Tweedie-denoised estimates of noised versions of input image x.
        """

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = normalize_np(x) * 255.
        x0 = np.transpose(x, (1, 2, 0)).astype(np.uint8)
        x0 = Image.fromarray(x0)
        x0 = T.Compose([T.Resize((256, 256)), T.ToTensor()])(x0)
        x0 = x0.unsqueeze(0)
        x0 = (x0.to(self.device).mul(2).sub(1).clone())

        x0ts, fx0ts = [], []
        for i in range(n_perturbations):
            x0t, xt = self.diffusion.tweedie_simple(
                model=self.diffusion_model,
                x0=x0,
                t=idx,
            )

            im = x0t.squeeze().detach().cpu().numpy()
            im = np.transpose(im, (1, 2, 0))
            im = normalize_np(im)
            im = (im * 255).astype(np.uint8)
            if self.transforms is not None:
                im = Image.fromarray(im)
                im = self.transforms(im)
            else:
                im = torch.from_numpy(im)
            pred = self.model(im.unsqueeze(0).to(self.device))
            pred = pred.squeeze().detach().cpu().numpy()
            # fx0t = pred[target_category]

            x0t = T.Resize((224, 224))(x0t)
            x0t = x0t.squeeze().detach().cpu().numpy()

            x0ts.append(x0t)
            fx0ts.append(pred)

        x0ts = np.array(x0ts)
        fx0ts = np.array(fx0ts)

        return x0ts, fx0ts

    def get_gfgp(self, fx, xps, fxps, category_ind):
        xps, fxps = xps.copy(), fxps.copy()

        xp_bar = np.mean(xps, axis=0)
        f_bar = np.mean(fxps, axis=0)

        ec = np.zeros_like(fx)
        ec[:, category_ind] = 1

        px = softmax(fx.squeeze())

        ecmp = (ec - px).squeeze()
        fmfbar = fxps - f_bar[np.newaxis, ...]
        xmxbar = xps - xp_bar[np.newaxis, ...]

        weight = fmfbar @ ecmp

        gfgp = np.sum(
            xmxbar * weight[:, np.newaxis, np.newaxis, np.newaxis], axis=0
        )

        return gfgp

    def attribute(self, inputs, targets):

        gfgps = []
        for x, y in zip(inputs, targets):
            gfgp_t = []
            for timestep in self.timesteps:
                fx = self.model(x.unsqueeze(0)).detach().cpu().numpy()
                x0ts, fx0ts = self.tweedie_perturb(
                    x, timestep, self.n_perturbations)
                gfgp = self.get_gfgp(fx, x0ts, fx0ts, y.item())
                gfgp_t.append(gfgp)

            gfgp = np.array(gfgp_t).mean(axis=0)
            gfgps.append(gfgp)
        gfgps = np.array(gfgps)
        gfgps = torch.Tensor(gfgps)

        return gfgps
