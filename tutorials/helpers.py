from typing import Optional, List
import os
import json
from pathlib import Path

import torchvision
from torch import Tensor
from torch.utils.data import Dataset, Subset
from PIL import Image


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'samples/')
        self.label_dir = os.path.join(self.root_dir, 'imagenet_class_index.json')
        
        with open(self.label_dir) as json_data:
            self.idx_to_labels = json.load(json_data)
        
        self.img_names = os.listdir(self.img_dir)
        self.img_names.sort()
        
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = idx
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def idx_to_label(self, idx):
        return self.idx_to_labels[str(idx)][1]


def get_torchvision_model(model_name):
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = torchvision.models.get_model(model_name, weights=weights).eval()
    transform = weights.transforms()
    return model, transform

def get_imagenet_dataset(
        transform,
        subset_size: int=100, # ignored if indices is not None
        root_dir="./data/ImageNet",
        indices: Optional[List[int]]=None,
    ):
    os.chdir(Path(__file__).parent) # ensure path
    dataset = ImageNetDataset(root_dir=root_dir, transform=transform)
    if indices is not None:
        return Subset(dataset, indices=indices)
    indices = list(range(len(dataset)))
    subset = Subset(dataset, indices=indices[:subset_size])
    return subset

img_to_np = lambda img: img.permute(1, 2, 0).detach().numpy()

def denormalize_image(inputs, mean, std):
    return img_to_np(
        inputs
        * Tensor(std)[:, None, None]
        + Tensor(mean)[:, None, None]
    )


def get_livertumor_dataset(
        transform,
        subset_size: int=100, # ignored if indices is not None
        root_dir="./data/LiverTumor",
        indices: Optional[List[int]]=None,
    ):
    from datasets.liver_tumor import LiverTumorDataset

    os.chdir(Path(__file__).parent) # ensure path
    dataset = LiverTumorDataset(data_dir=root_dir, transform=transform)
    if indices is not None:
        return Subset(dataset, indices=indices)
    indices = list(range(len(dataset)))
    subset = Subset(dataset, indices=indices[:subset_size])
    return subset


def get_livertumor_model(model_dir):
    from models.liver_tumor import ResNet50
    import torch
    from torchvision import transforms
    model = ResNet50(in_channels=1, num_classes=2)
    # checkpoint = torch.load(model_dir)['net']
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint, strict=False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    return model, transform

sample_to_np = lambda sample: sample.permute(1, 2, 0).squeeze(dim=-1).detach().numpy()

def denormalize_sample(inputs, mean, std):
    return sample_to_np(
        inputs
        * Tensor(std)
        + Tensor(mean)
    )
