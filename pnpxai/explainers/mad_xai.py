import os
import sys
from typing import Optional, Tuple, Any

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from pnpxai.explainers.base import Explainer


def smooth_mask(mask, sigma=1.0):
    """Apply Gaussian smoothing to anomaly mask"""
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


DEFAULT_REPO = "kaistpnp/madical_pnpxai"
DEFAULT_CONFIG = {
    'model': 'UNet_M',
    'modality': 't2',
    'image_size': 256,
    'ddim_steps': 10,
    'vae_repo': 'farzadbz/Medical-VAE',
    'vae_filename': 'VAE-Medical-klf8.pt',
}


def get_default_model_and_diffusion(
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    hf_repo: Optional[str] = None
) -> Tuple[torch.nn.Module, Any, Any]:
    """
    Load MAD-XAI model, VAE, and diffusion from HuggingFace Hub.
    
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
    
    # Download entire repository (uses cache if already downloaded)
    # This allows us to import models.py and diffusion_x0 directly
    print(f"📥 Loading repository {hf_repo}...")
    repo_dir = snapshot_download(hf_repo)
    
    # Add repository directory to Python path to import modules
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    
    # Load model checkpoint
    model_filename = config.get('model_filename', 'MAD_t2_UNet_M/checkpoints/last.pt')
    print(f"📥 Loading model from {hf_repo}/{model_filename}...")
    model_path = hf_hub_download(
        repo_id=hf_repo,
        filename=model_filename,
        cache_dir=None
    )
    
    # Load config file
    try:
        config_path = hf_hub_download(
            repo_id=hf_repo,
            filename="MAD_t2_UNet_M/args.yml",
            cache_dir=None
        )
        with open(config_path, 'r') as f:
            hf_config = yaml.safe_load(f)
        model_name = hf_config.get('model', config['model'])
        config.update(hf_config)
    except Exception:
        model_name = config['model']
        print("⚠ Config file not found, using defaults")
    
    # Import models.py from repository (now available in sys.path)
    try:
        from models import UNET_models
        print(f"✓ Loaded models.py from repository")
    except ImportError as e:
        print(f"⚠ Could not import UNET_models: {e}")
        print(f"⚠ Repository directory: {repo_dir}")
        raise ImportError(
            f"Could not import UNET_models from {hf_repo}. "
            "Please ensure models.py exists in the repository."
        )
    
    # Initialize model
    in_channels = out_channels = 4
    unet_model = UNET_models[model_name](in_channels=in_channels, out_channels=out_channels)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    unet_model.load_state_dict(state_dict)
    unet_model.eval()
    unet_model.to(device)
    
    # Import ldm module from repository (needed for VAE loading)
    # VAE model was saved with ldm classes, so we need to import ldm before loading
    print(f"📥 Loading ldm module from repository...")
    try:
        import ldm
        print(f"✓ Loaded ldm module from repository")
    except ImportError as e:
        print(f"⚠ Could not import ldm module: {e}")
        print(f"⚠ Repository directory: {repo_dir}")
        # Check if ldm directory exists in repo
        import os
        ldm_path = os.path.join(repo_dir, 'ldm')
        if os.path.exists(ldm_path):
            print(f"⚠ ldm directory exists at {ldm_path}, but import failed")
        else:
            print(f"⚠ ldm directory not found in repository")
        raise ImportError(
            f"Could not import ldm module from {hf_repo}. "
            "Please ensure ldm module exists in the repository and is properly structured."
        )
    
    # Load VAE
    vae_repo = config.get('vae_repo', DEFAULT_CONFIG['vae_repo'])
    vae_filename = config.get('vae_filename', DEFAULT_CONFIG['vae_filename'])
    print(f"📥 Downloading VAE from {vae_repo}/{vae_filename}...")
    vae_model_path = hf_hub_download(
        repo_id=vae_repo,
        filename=vae_filename
    )
    vae = torch.load(vae_model_path, map_location=device)
    vae.eval()
    vae.to(device)
    
    # Import diffusion_x0 from repository (now available in sys.path)
    try:
        from diffusion_x0 import create_diffusion
        print(f"✓ Loaded diffusion_x0 from repository")
    except ImportError as e:
        print(f"⚠ Could not import create_diffusion: {e}")
        print(f"⚠ Repository directory: {repo_dir}")
        raise ImportError(
            f"Could not import diffusion_x0 from {hf_repo}. "
            "Please ensure diffusion_x0 module exists in the repository."
        )
    
    ddim_steps = config.get('ddim_steps', DEFAULT_CONFIG['ddim_steps'])
    diffusion = create_diffusion(
        timestep_respacing=f"ddim{ddim_steps}",
        predict_xstart=True,
        sigma_small=False,
        learn_sigma=False,
        diffusion_steps=10
    )
    
    print(f"✓ Model loaded successfully")
    return unet_model, vae, diffusion


class MadXai(Explainer):
    """
    MAD-XAI (Medical Anomaly Detection using Diffusion) Explainer.
    
    This explainer uses diffusion models for medical image anomaly detection.
    It reconstructs input images using a diffusion model and computes anomaly
    maps based on the difference between original and reconstructed images.
    
    Parameters:
        model: The UNet diffusion model (optional, will be loaded from HF if not provided)
        vae: The VAE model for encoding/decoding (optional, will be loaded from HF if not provided)
        diffusion: The diffusion sampler (optional, will be created if not provided)
        ddim_steps: Number of DDIM sampling steps
        model_config: Configuration dict for model loading (overrides DEFAULT_CONFIG)
        model_device: Device to load models on
        forward_arg_extractor: Optional function to extract forward arguments
        additional_forward_arg_extractor: Optional function for additional forward args
    """
    
    def __init__(
        self,
        model=None,  # UNet diffusion model
        vae=None,  # VAE model
        diffusion=None,  # Diffusion sampler
        ddim_steps: int = 10,
        model_config: Optional[dict] = None,
        model_device: Optional[torch.device] = None,
        forward_arg_extractor=None,
        additional_forward_arg_extractor=None,
        **kwargs,
    ):
        # For MAD-XAI, we don't use a classification model in the traditional sense
        # Instead, we use the diffusion model for reconstruction
        # We'll create a dummy model for the base class
        if model is None:
            # Create a dummy model that just passes through
            class DummyModel(torch.nn.Module):
                def forward(self, x):
                    return x
            dummy_model = DummyModel()
        else:
            dummy_model = model
        
        super().__init__(dummy_model, forward_arg_extractor, additional_forward_arg_extractor)
        
        model_device = model_device if model_device is not None else self.device
        
        # Load models if not provided
        if model is None or vae is None or diffusion is None:
            unet_model, vae_model, diffusion_obj = get_default_model_and_diffusion(
                model_config, model_device
            )
            self.unet_model = unet_model
            self.vae = vae_model
            self.diffusion = diffusion_obj
        else:
            self.unet_model = model
            self.vae = vae
            self.diffusion = diffusion
        
        self.ddim_steps = ddim_steps
        self.model_device = model_device

    def compute_anomaly_map(self, x0, x_reconstructed):
        """
        Compute anomaly map from original and reconstructed images.
        
        Args:
            x0: Original image tensor [B, C, H, W] or [C, H, W]
            x_reconstructed: Reconstructed image tensor [B, C, H, W] or [C, H, W]
            
        Returns:
            Anomaly map as numpy array [H, W] or [B, H, W] if batch input
        """
        # Handle single image vs batch
        if x0.dim() == 3:
            x0 = x0.unsqueeze(0)
            x_reconstructed = x_reconstructed.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Compute absolute difference per image
        anomaly_maps = []
        for x0_img, x_rec_img in zip(x0, x_reconstructed):
            image_difference = (
                torch.abs(x_rec_img - x0_img)
                .to(torch.float32)
                .detach()
                .cpu()
                .numpy()
            )
            
            # Handle multi-channel: take max across channels
            if len(image_difference.shape) == 3:
                # [C, H, W] -> [H, W] by taking mean across channels
                image_difference = image_difference.mean(axis=0)
            elif len(image_difference.shape) == 2:
                pass  # Already 2D
            else:
                raise ValueError(f"Unexpected image_difference shape: {image_difference.shape}")
            
            # Clip and normalize
            image_difference = np.clip(image_difference, 0.0, 0.25) * 4
            
            # Apply smoothing
            final_anomaly = smooth_mask(image_difference, sigma=5)
            anomaly_maps.append(final_anomaly)
        
        if squeeze_output:
            return anomaly_maps[0]
        else:
            return np.stack(anomaly_maps, axis=0)

    def attribute(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute anomaly detection attributions for input images.
        
        Args:
            inputs: Input image tensor [B, C, H, W] or [C, H, W]
            targets: Not used for anomaly detection (kept for API compatibility)
            
        Returns:
            Anomaly maps as tensor [B, H, W] or [H, W]
        """
        # Handle single image vs batch
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        inputs = inputs.to(self.model_device)
        
        anomaly_maps = []
        
        with torch.no_grad():
            for x in inputs:
                x = x.unsqueeze(0)  # Add batch dimension
                
                # Encode to latent space
                encoded = self.vae.encode(x).mean.mul_(0.18215)  # Normalization from LDM
                
                # Reconstruct using diffusion
                model_kwargs = {'mask': torch.ones_like(encoded)}
                latent_corrected = self.diffusion.ddim_sample_loop(
                    self.unet_model,
                    encoded.shape,
                    noise=encoded,
                    clip_denoised=False,
                    t=self.ddim_steps,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=self.model_device,
                    eta=0.5
                )
                
                # Decode reconstructed latent
                x_reconstructed = self.vae.decode(latent_corrected / 0.18215)
                
                # Decode original latent for comparison
                x0 = self.vae.decode(encoded / 0.18215)
                
                # Compute anomaly map
                anomaly_map = self.compute_anomaly_map(x0, x_reconstructed)
                anomaly_maps.append(anomaly_map)
        
        anomaly_maps = np.stack(anomaly_maps, axis=0)
        anomaly_maps = torch.from_numpy(anomaly_maps).float()
        
        if squeeze_output:
            anomaly_maps = anomaly_maps.squeeze(0)
        
        return anomaly_maps

