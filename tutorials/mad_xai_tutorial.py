#%%
"""
MAD-XAI Tutorial: Medical Anomaly Detection using Diffusion Models

This tutorial demonstrates how to use the MAD-XAI explainer for medical image
anomaly detection. The explainer uses diffusion models to reconstruct images
and identifies anomalies based on reconstruction errors.

Install the following packages before running:
pip install opencv-python torchmetrics einops lightning taming-transformers albumentations

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import zipfile
from pathlib import Path
from scipy.ndimage import gaussian_filter
import cv2
from PIL import Image
import copy
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from torchmetrics.classification import BinaryAUROC, BinaryPrecisionRecallCurve

from pnpxai.explainers import MadXai
from huggingface_hub import hf_hub_download, snapshot_download
import sys

#%%
# Helper functions for evaluation (specific to this tutorial)

def smooth_mask(mask, sigma=1.0):
    """Apply Gaussian smoothing to mask"""
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


def compute_dice(anomaly_map, segmentation, th):
    """Compute Dice score at given threshold"""
    anomaly_map = copy.deepcopy(anomaly_map)
    anomaly_map[anomaly_map > th] = 1
    anomaly_map[anomaly_map < 1] = 0
    
    if sum(segmentation.flatten()) == 0:
        if sum(anomaly_map.flatten()) == 0:
            return 1
        else:
            return 0
    
    anomaly_map[anomaly_map < 1] = 0
    eps = 1e-6
    inputs = anomaly_map.flatten()
    targets = segmentation.flatten()
    
    intersection = (inputs * targets).sum()
    dice = (2. * intersection) / (inputs.sum() + targets.sum() + eps)
    return dice


def dsc_max(anomaly_maps, segmentations):
    """Find maximum Dice score across thresholds"""
    dice_scores = []
    ths = np.linspace(0, 1, 201)
    best_dsc = 0
    outer_pbar = tqdm(ths, desc="Searching best threshold")
    for dice_threshold in outer_pbar:
        dice_scores = []
        for k in tqdm(range(len(anomaly_maps)), desc=f"Computing dice (threshold={dice_threshold:.3f})", leave=False):
            dice = compute_dice(
                copy.deepcopy(np.asarray(anomaly_maps[k]).flatten()),
                copy.deepcopy(np.asarray(segmentations[k]).flatten()),
                dice_threshold
            )
            dice_scores.append(dice)
        mean_dice = np.mean(dice_scores)
        if mean_dice > best_dsc:
            best_dsc = mean_dice
        outer_pbar.set_postfix({'current_dice': f'{mean_dice:.4f}', 'best_dice': f'{best_dsc:.4f}'})
    return best_dsc


def calculate_metrics(ground_truth, prediction):
    """Calculate evaluation metrics"""
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    
    max_dicescore = dsc_max(prediction, ground_truth)
    
    # Convert to Tensor once to reuse
    flat_pred_tensor = torch.from_numpy(flat_pred)
    flat_gt_tensor = torch.from_numpy(flat_gt.astype(int))
    
    auroc = BinaryAUROC()
    auroc_score = auroc(flat_pred_tensor, flat_gt_tensor)
    
    pr_curve = BinaryPrecisionRecallCurve()
    pr_curve.update(flat_pred_tensor, flat_gt_tensor)
    
    precision, recall, thresholds = pr_curve.compute()
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    f1_max_score = torch.max(f1_scores)
    
    ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    return auroc_score.cpu().numpy(), f1_max_score.cpu().numpy(), ap, max_dicescore


def visualize(anomaly_maps, segmentations, xs, image_samples, image_size):
    # Limit to first 4 images only
    num_visualize = min(4, len(anomaly_maps))
    
    # Prepare images for 4x4 grid (4 rows x 4 columns)
    # Each row represents one image set: [input, output, segmentation, anomaly_map]
    all_images = []
    
    for anomaly_map, segmentation, x, image_sample in zip(anomaly_maps[:num_visualize], segmentations[:num_visualize], xs[:num_visualize], image_samples[:num_visualize]):
        # Convert tensors to images
        input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1, 2, 0)) * 127.5 + 127.5).astype(np.uint8)
        output_image = ((np.clip(image_sample[0].detach().cpu().numpy(), -1, 1).transpose(1, 2, 0)) * 127.5 + 127.5).astype(np.uint8)
        
        # Create anomaly map visualization
        # Ensure anomaly_map is 2D and properly normalized
        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()
        
        # Squeeze extra dimensions to ensure 2D shape (H, W)
        anomaly_map = np.squeeze(anomaly_map)
        if anomaly_map.ndim != 2:
            raise ValueError(f"Expected 2D anomaly_map, got shape {anomaly_map.shape}")
        
        # Normalize to [0, 1] range
        anomaly_map_min = anomaly_map.min()
        anomaly_map_max = anomaly_map.max()
        if anomaly_map_max > anomaly_map_min:
            anomaly_map = (anomaly_map - anomaly_map_min) / (anomaly_map_max - anomaly_map_min)
        else:
            anomaly_map = np.zeros_like(anomaly_map)
        
        # Convert to uint8 and apply colormap
        anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)[:, :, ::-1]
        anomal_map_img = (0.5 * input_image + (1 - 0.5) * scoremap).astype(np.uint8)
        
        # Create segmentation visualization
        seg_img = 255 * np.repeat(segmentation.cpu().numpy(), 3, axis=0).transpose([1, 2, 0])
        
        # Store images for this row: [input, output, segmentation, anomaly_map]
        all_images.append([input_image, output_image, seg_img, anomal_map_img])
    
    # Create 4x4 grid visualization
    # Each row has 4 images horizontally arranged
    grid_image = np.zeros((num_visualize * image_size, 4 * image_size, 3)).astype(np.uint8)
    
    for row_idx, images in enumerate(all_images):
        input_img, output_img, seg_img, anomal_img = images
        start_row = row_idx * image_size
        end_row = (row_idx + 1) * image_size
        
        grid_image[start_row:end_row, :image_size] = input_img
        grid_image[start_row:end_row, image_size:2*image_size] = output_img
        grid_image[start_row:end_row, 2*image_size:3*image_size] = seg_img
        grid_image[start_row:end_row, 3*image_size:] = anomal_img
    
    # Plot visualization using matplotlib with reduced figure size
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(grid_image)
    ax.axis('off')
    ax.set_title('Anomaly Detection Results (4x4 Grid)')
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def load_model_from_hub(repo_id, filename="MAD_t2_UNet_M/checkpoints/last.pt", device="cuda"):
    """Load model from HuggingFace Hub (tutorial-specific)"""
    # Download entire repository to get models module
    print(f"📥 Downloading repository {repo_id}...")
    repo_dir = snapshot_download(repo_id)
    # Add repository directory to Python path to import models
    sys.path.insert(0, repo_dir)
    
    # Try to find models.py in the repository
    import os
    models_path = None
    for root, dirs, files in os.walk(repo_dir):
        if 'models.py' in files:
            models_path = root
            break
    
    if models_path:
        sys.path.insert(0, models_path)
        print(f"✓ Found models.py at {models_path}")
    else:
        print(f"⚠ models.py not found in repository, trying default import from {repo_dir}")
    
    print(f"📥 Downloading model from {repo_id}/{filename}...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None
    )
    
    # Download config file
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="MAD_t2_UNet_M/args.yml",
            cache_dir=None
        )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config['model']
        print(f"✓ Loaded config: model={model_name}, modality={config['modality']}, image_size={config['image_size']}")
    except:
        config = {'model': 'UNet_M', 'modality': 't2', 'image_size': 256}
        model_name = 'UNet_M'
        print("⚠ Config file not found, using defaults")
    
    # Import model (now available from downloaded repository)
    try:
        from models import UNET_models
    except ImportError as e:
        print(f"⚠ Import error: {e}")
        print(f"⚠ Repository directory: {repo_dir}")
        print(f"⚠ Files in repository root: {os.listdir(repo_dir) if os.path.exists(repo_dir) else 'N/A'}")
        raise ImportError(
            f"Could not import UNET_models. Repository downloaded to {repo_dir}. "
            "Please check if models.py exists in the repository."
        )
    
    # Initialize model
    in_channels = out_channels = 4
    model = UNET_models[model_name](in_channels=in_channels, out_channels=out_channels)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    print(f"✓ Model loaded successfully")
    return model, config


def load_dataset_from_hub(repo_id, split="test", local_dir="./data_hf", modality="t2", image_size=256):
    """Load dataset from HuggingFace Hub (tutorial-specific)"""
    print(f"📥 Downloading {split}.zip from {repo_id}...")
    
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Download zip
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{split}.zip",
        repo_type="dataset",
        cache_dir=None
    )
    
    # Extract zip
    print(f"📦 Extracting {split}.zip to {local_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_dir)
    
    print(f"✓ Dataset extracted successfully")
    
    # Create dataset (assuming MedicalDataset is available)
    try:
        from MedicalDataLoader import MedicalDataset
    except ImportError:
        raise ImportError("Please ensure MedicalDataLoader module is available in Python path")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    
    dataset = MedicalDataset(
        mode=split,
        rootdir=str(local_dir),
        transform=transform,
        image_size=image_size,
        augment=False,
        modality=modality
    )
    
    return dataset


#%%
# Main tutorial execution

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("MAD-XAI Tutorial: Medical Anomaly Detection")
    print("="*60)
    
    # Configuration
    model_repo_id = "kaistpnp/madical_pnpxai"
    model_filename = "MAD_t2_UNet_M/checkpoints/last.pt"
    dataset_repo_id = "kaistpnp/madical_pnpxai_brats"
    data_root = "./data_hf"
    output_dir = "./results"
    batch_size = 16
    ddim_steps = 10
    num_workers = 4
    
    # Load model and config
    model, config = load_model_from_hub(
        repo_id=model_repo_id,
        filename=model_filename,
        device=device
    )
    
    # Load VAE
    print(f"📥 Downloading VAE from farzadbz/Medical-VAE...")
    vae_model_path = hf_hub_download(repo_id="farzadbz/Medical-VAE", filename="VAE-Medical-klf8.pt")
    vae = torch.load(vae_model_path, map_location=device)
    vae.eval()
    vae.to(device)
    print("✓ VAE loaded successfully")
    
    # Load dataset
    test_dataset = load_dataset_from_hub(
        repo_id=dataset_repo_id,
        split="test",
        local_dir=data_root,
        modality=config.get('modality', 't2'),
        image_size=config.get('image_size', 256)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    print(f"✓ Dataset contains {len(test_dataset)} test images")
    
    # Set parameters
    image_size = config.get('image_size', 256)
    modality = config.get('modality', 't2')
    
    # Create diffusion
    from diffusion_x0 import create_diffusion
    diffusion = create_diffusion(
        timestep_respacing=f"ddim{ddim_steps}",
        predict_xstart=True,
        sigma_small=False,
        learn_sigma=False,
        diffusion_steps=10
    )
    
    # Initialize MAD-XAI explainer
    explainer = MadXai(
        model=model,
        vae=vae,
        diffusion=diffusion,
        ddim_steps=ddim_steps,
        model_device=device
    )
    
    # Run inference
    print('=-='*20)
    print('Starting evaluation...')
    print('=-='*20)
    
    image_samples_s = []
    x0_s = []
    segmentation_s = []
    
    for ii, (x, brain_mask, seg) in enumerate(test_loader):
        with torch.no_grad():
            # Use explainer to compute anomaly maps
            # Note: For full pipeline, we also need reconstructed images
            # This is a simplified version - full version would compute both
            
            # Encode to latent space
            encoded = vae.encode(x.to(device)).mean.mul_(0.18215)
            
            # Reconstruct using diffusion
            model_kwargs = {'mask': torch.ones_like(encoded)}
            latent_corrected = diffusion.ddim_sample_loop(
                model, encoded.shape, noise=encoded, clip_denoised=False,
                t=ddim_steps,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
                eta=0.5
            )
            
            image_samples = vae.decode(latent_corrected / 0.18215)
            x0 = vae.decode(encoded / 0.18215)
            
            segmentation_s += [_seg.unsqueeze(0) for _seg in seg]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]
    
    # Compute anomaly maps using explainer
    anomaly_maps_list = []
    for x0_img, image_sample_img in zip(x0_s, image_samples_s):
        # x0_img and image_sample_img are already single image tensors [1, C, H, W]
        anomaly_map = explainer.compute_anomaly_map(x0_img, image_sample_img)
        if isinstance(anomaly_map, np.ndarray):
            anomaly_maps_list.append(anomaly_map)
        else:
            # If it's already a list/array, extend
            anomaly_maps_list.extend(anomaly_map if isinstance(anomaly_map, list) else [anomaly_map])
    
    # Prepare ground truth
    gt = []
    for seg in segmentation_s:
        gt.append(seg[0, :, :].cpu().numpy())
    
    # Stack anomaly maps
    if len(anomaly_maps_list) > 0 and isinstance(anomaly_maps_list[0], np.ndarray):
        anomaly_maps = np.stack(anomaly_maps_list, axis=0)
    else:
        anomaly_maps = np.array(anomaly_maps_list)
    gt = np.stack(gt, axis=0)
    gt = (gt > 0).astype(np.int32)
    
    # Calculate metrics
    auroc_score, f1_max_score, ap, max_dicescore = calculate_metrics(gt, anomaly_maps)
    
    print('Results:')
    print(f'max Dice score: {max_dicescore:.4f}')
    print(f'F1 score: {f1_max_score:.4f}')
    print(f'AUROC: {auroc_score:.4f}')
    print(f'AP: {ap:.4f}')
    
    # Visualize results
    visualize(anomaly_maps, segmentation_s, x0_s, image_samples_s, image_size)


if __name__ == "__main__":
    main()

# %%

