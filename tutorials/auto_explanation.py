import os
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision.models.resnet import resnet18, ResNet18_Weights

from pnpxai.utils import set_seed
from pnpxai import Project

from helpers import get_imagenet_dataset, get_torchvision_model


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

set_seed(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#-----------------------------------------------------------------------------#
#----------------------------------- model -----------------------------------#
#-----------------------------------------------------------------------------#

model, transform = get_torchvision_model("resnet18")
model = model.to(device)


#-----------------------------------------------------------------------------#
#------------------------------------ data -----------------------------------#
#-----------------------------------------------------------------------------#

dataset = get_imagenet_dataset(transform, subset_size=100)
loader = DataLoader(dataset, batch_size=8)


#-----------------------------------------------------------------------------#
#---------------------------------- explain ----------------------------------#
#-----------------------------------------------------------------------------#

proj = Project('test_project')
expr = proj.create_experiment(model, auto=True)
for inputs, targets in loader:
    expr.run(inputs, targets)

print(len(expr.runs))