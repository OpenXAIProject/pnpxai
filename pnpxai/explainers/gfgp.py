from .base import Explainer
from .blended_diffusion_custom.guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
import os
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

def download(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

class Gfgp(Explainer):
    def __init__(self,
                 model, # classification model to be explained
                 transforms,
                 
                 model_config=None,
                 diffusion_ckpt_path=None,
                 forward_arg_extractor = None,
                 additional_forward_arg_extractor = None,
    ):
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        
        self.device = torch.device('cuda')
        self.transforms = transforms
        
        if model_config is not None:
            self.model_config = model_config
        else:
            self.model_config = {'image_size': 256, # 256
                'num_channels' : 256, 
                'num_res_blocks': 2,
                'resblock_updown': True,
                'num_heads': 4,
                'num_heads_upsample': -1,
                'num_head_channels': 64,
                'attention_resolutions' : '32, 16, 8',
                'channel_mult': '',
                'dropout': 0.0,
                'class_cond': False,
                'use_checkpoint': False,
                'use_scale_shift_norm': True,
                'use_fp16': True, # True,
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
        self.diffusion_model, self.diffusion = create_model_and_diffusion(**self.model_config)
        if diffusion_ckpt_path is None:
            print("No checkpoint path for diffusion model was specified.\n Using default diffusion model from OpenAI's guided-diffusion repository at\n https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt")
            url = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
            filename = "256x256_diffusion_uncond.pt"
            download(url, filename)
            diffusion_ckpt_path = "256x256_diffusion_uncond.pt"
        
        self.diffusion_model.load_state_dict(torch.load(diffusion_ckpt_path, map_location="cpu"))
        self.diffusion_model.requires_grad_(False).eval().to(self.device)
        if self.model_config["use_fp16"]:
            self.diffusion_model.convert_to_fp16()
        
        self.model.to(self.device)
        
        self.index = 300
        self.n_perturbations = 100
        
        
    def tweedie_perturb(self, x, target_category, index=300, n_perturbations=100):
        
        """
        Generate perturbations using diffusion
        The perturbations are Tweedie-denoised estimates of noised versions of input image x.
        """
        
        # fx = self.model(torch.tensor(x).unsqueeze(0).to('cuda'))[0].detach().cpu()
        
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = normalize_np(x) * 255.
        x0 = np.transpose(x, (1,2,0)).astype(np.uint8)
        x0 = Image.fromarray(x0)
        x0 = T.Compose([T.Resize((256,256)),T.ToTensor()])(x0)
        x0 = x0.unsqueeze(0)
        x0 = (x0.to(self.device).mul(2).sub(1).clone())

        x0ts, fx0ts = [], []        
        for i in range(n_perturbations):
            x0t, xt = self.diffusion.tweedie_simple(
                model = self.diffusion_model,
                x0=x0,
                t=index,
            )
            
            im = x0t.squeeze().detach().cpu().numpy()
            im = np.transpose(im, (1,2,0))
            im = normalize_np(im)
            im = (im * 255).astype(np.uint8)
            im = Image.fromarray(im)
            im = self.transforms(im)
            pred = self.model(im.unsqueeze(0).to(self.device))
            pred = pred.squeeze().detach().cpu().numpy()
            fx0t = pred[target_category]
            
            x0t = T.Resize((224,224))(x0t)
            x0t = x0t.squeeze().detach().cpu().numpy()
            
            x0ts.append(x0t)
            fx0ts.append(fx0t)
        
        x0ts = np.array(x0ts)
        fx0ts = np.array(fx0ts)
            
        return x0ts, fx0ts
    
    def get_gfgp(self, x0ts, fx0ts):
        x0ts, fx0ts = x0ts.copy(), fx0ts.copy()
        
        E_x0t = np.mean(x0ts, axis=0)
        xx = E_x0t - x0ts
        xx = normalize_np(xx)
        
        E_fx0t = np.mean(fx0ts)
        ff = E_fx0t - fx0ts
        
        ffxx = ff[:, None, None, None] * xx
        
        gfgp = np.mean(ffxx, axis=0)
        gfgp -= gfgp.mean()
        
        return gfgp
    
    def attribute(self, inputs, targets):
        
        gfgps = []
        for i, (x, y) in enumerate(zip(inputs, targets)):
            x0ts, fx0ts = self.tweedie_perturb(x, y, self.index, self.n_perturbations)
            gfgp = self.get_gfgp(x0ts, fx0ts)
            gfgps.append(gfgp)
        gfgps = np.array(gfgps)
        gfgps = torch.Tensor(gfgps)
        
        return gfgps