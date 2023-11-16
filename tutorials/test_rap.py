import os
import json
from typing import List, Optional
import torch
from torch import fx
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from torchvision.models.resnet import resnet18, ResNet18_Weights, BasicBlock, Bottleneck

from pnpxai.explainers import RAP
from pnpxai.explainers.utils.operation_graph import OperationGraph
from pnpxai.explainers.rap.rules import RelPropSimple


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'samples/')
        self.label_dir = os.path.join(
            self.root_dir, 'imagenet_class_index.json')

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


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

imagenet_data = ImageNetDataset(
    root_dir='./data/ImageNet/', transform=data_transforms)
imagenet_data_loader = DataLoader(
    imagenet_data,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
rap = RAP(resnet)
rap.attribute(imagenet_data[0][0].unsqueeze(0), imagenet_data[0][1])
# rap.method.graph.pprint()