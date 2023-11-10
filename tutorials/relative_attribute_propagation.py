from functools import partial

import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg16_bn, VGG16_BN_Weights
from torchvision.models.resnet import resnet18, ResNet18_Weights, BasicBlock, Bottleneck
import torchvision.transforms as transforms

from pnpxai import Project
from pnpxai.explainers import Explainer, RAP
from pnpxai.explainers.rap.rules import RelPropSimple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(path: str) -> DataLoader:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return DataLoader(
        datasets.ImageFolder(path, transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )


def vgg_relprop(self, r: torch.Tensor):
    x1 = self.model.classifier.rule.relprop(r)
    if torch.is_tensor(x1) == False:
        for i in range(len(x1)):
            x1[i] = x1[i].reshape_as(
                next(reversed(self.model.features._modules.values())).rule.Y)
    else:
        x1 = x1.reshape_as(
            next(reversed(self.model.features._modules.values())).rule.Y)

    x1 = self.model.avgpool.rule.relprop(x1)
    x1 = self.model.features.rule.relprop(x1)

    return x1


def explain_vgg(project: Project, data_loader: DataLoader):
    model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).to(device)
    explainer = RAP(model)
    explainer.method.relprop = partial(vgg_relprop, explainer.method)

    exp = project.explain(explainer)
    exp.run(data_loader)

    outputs = [output.detach().cpu() for output in exp.runs[-1].outputs]

    return exp, outputs


def resnet_relprop(self, r: torch.Tensor):
    r = self.model.fc.rule.relprop(r)
    r = r.reshape_as(self.model.avgpool.rule.Y)
    r = self.model.avgpool.rule.relprop(r)

    r = self.model.layer4.rule.relprop(r)
    r = self.model.layer3.rule.relprop(r)
    r = self.model.layer2.rule.relprop(r)
    r = self.model.layer1.rule.relprop(r)

    r = self.model.maxpool.rule.relprop(r)
    r = self.model.relu.rule.relprop(r)
    r = self.model.bn1.rule.relprop(r)
    r = self.model.conv1.rule.relprop(r)

    return r


def resnet_basic_block_relprop(self, r: torch.Tensor):
    x = self.downsample.rule.Y if self.downsample else self.conv1.rule.X_orig
    out = self.relu.rule.relprop(r)
    out, x2 = RelPropSimple().relprop(out, [self.bn2.rule.Y, x], self.relu.rule.X_orig)

    if self.downsample is not None:
        x2 = self.downsample.rule.relprop(x2)

    out = self.bn2.rule.relprop(out)
    out = self.conv2.rule.relprop(out)

    out = self.bn1.rule.relprop(out)
    x1 = self.conv1.rule.relprop(out)

    return x1 + x2


def resnet_bottleneck_relprop(self, r: torch.Tensor):
    x = self.downsample.rule.X_orig if self.downsample else self.conv1.rule.X_orig
    out = self.relu3.rule.relprop(r)

    out, x = RelPropSimple().relprop(out, [self.bn3.rule.Y, x])

    if self.downsample is not None:
        x = self.downsample.rule.relprop(x)

    out = self.bn3.rule.relprop(out)
    out = self.conv3.rule.relprop(out)

    out = self.relu2.rule.relprop(out)
    out = self.bn2.rule.relprop(out)
    out = self.conv2.rule.relprop(out)

    out = self.relu1.rule.relprop(out)
    out = self.bn1.rule.relprop(out)
    x1 = self.conv1.rule.relprop(out)

    return x1 + x


def explain_resnet(project: Project, data_loader: DataLoader):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    explainer = RAP(model)

    for layer in explainer.method.layers:
        if isinstance(layer, BasicBlock):
            layer.relprop = partial(resnet_basic_block_relprop, layer)

        if isinstance(layer, Bottleneck):
            layer.relprop = partial(resnet_bottleneck_relprop, layer)

    explainer.method.relprop = partial(resnet_relprop, explainer.method)

    exp = project.explain(explainer)
    exp.run(data_loader)

    outputs = [output.detach().cpu() for output in exp.runs[-1].outputs]

    return exp, outputs


def visualize(explainer: Explainer, inputs, outputs, path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    visualizations = explainer.format_outputs_for_visualization(inputs, outputs) or []
    for idx, visualization in enumerate(visualizations):
        visualization.write_image(f"{path}/rap_{idx}.png")


def app():
    data_path = './data/imagenet/'
    data_loader = load_data(data_path)

    project = Project('test_project')
    resnet_exp, resnet_outputs = explain_resnet(project, data_loader)

    visualizaion_path = f"./results/rap/resnet"
    visualize(resnet_exp.explainer, data_loader, resnet_outputs, visualizaion_path)

    vgg_exp, vgg_outputs = explain_vgg(project, data_loader)

    visualizaion_path = f"./results/rap/vgg"
    visualize(vgg_exp.explainer, data_loader, vgg_outputs, visualizaion_path)


if __name__ == '__main__':
    app()
