import os
import json

from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.vgg import vgg16_bn, VGG16_BN_Weights
from torchvision.models.resnet import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

from pnpxai.explainers import Explainer, RAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def load_data(path: str) -> DataLoader:
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    imagenet_data = ImageNetDataset(root_dir=path, transform=data_transforms)

    return DataLoader(
        imagenet_data,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )


def explain_model(model: nn.Module, data_loader: DataLoader):
    explainer = RAP(model)
    outputs = [
        explainer.attribute(inputs, labels).detach().cpu()
        for inputs, labels in data_loader
    ]

    return explainer, outputs


def visualize(explainer: Explainer, inputs, outputs, path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    visualizations = [
        explainer.format_outputs_for_visualization(batch_in, batch_out)
        for batch_in, batch_out in zip(inputs, outputs)
    ] or [[]]
    for batch_idx, batch_viz in enumerate(visualizations):
        for idx, visualization in enumerate(batch_viz):
            visualization.write_image(f"{path}/rap_{batch_idx}_{idx}.png")


def app():
    data_path = './data/ImageNet/'
    data_loader = load_data(data_path)

    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    explainer, resnet_outputs = explain_model(resnet, data_loader)

    visualizaion_path = f"./results/rap/resnet"
    visualize(explainer, data_loader, resnet_outputs, visualizaion_path)

    model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).to(device)
    explainer, vgg_outputs = explain_model(model, data_loader)

    visualizaion_path = f"./results/rap/vgg"
    visualize(explainer, data_loader, vgg_outputs, visualizaion_path)


if __name__ == '__main__':
    app()
