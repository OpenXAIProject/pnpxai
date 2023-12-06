import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models

from open_xai import Project
from open_xai.explainers import IntegratedGradients


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
model.load_state_dict(torch.load('models/cifar_torchvision.pt'))

dataiter = iter(testloader)
images, labels = next(dataiter)

outputs = model(images)

_, predicted = torch.max(outputs, 1)

idx = 3

inputs = images[idx].unsqueeze(0)
inputs.requires_grad = True

model.eval()


project = Project('test_project')
exp = project.explain(IntegratedGradients(model))
run = exp.run(inputs, target=labels[idx], baselines=inputs * 0, return_convergence_delta=True)

plots = exp.visualize()

plot_path = 'results/ig'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    
for run_idx, run_plots in enumerate(plots):
    for idx, plot in enumerate(run_plots):
        plot.write_image(f"{plot_path}/ig_{run_idx}_{idx}.png")
