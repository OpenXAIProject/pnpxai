import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from zennit.composites import EpsilonPlus
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from pnpxai.explainers import LRP
from pnpxai.explainers.utils.post_process import postprocess_attr

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


dataset_size = 100
batch_size = 32
assert dataset_size >= batch_size, "dataset_size should be larger than batch_size."

savedir = './results/lrp_vit'
os.makedirs(savedir, exist_ok=True)

model, transform = get_torchvision_model("vit_b_16")
dataset = get_imagenet_dataset(transform, subset_size=dataset_size)
loader = DataLoader(dataset, batch_size=batch_size)

explainer = LRP(model)

for i, data in enumerate(loader):
    inputs, labels = data
    targets = model(inputs).argmax(1)

    # attribute
    attributions = explainer.attribute(inputs, targets, epsilon=1e-1)
    attributions = postprocess_attr(attributions, sign='positive')

    # visualize
    for idx, attribution in enumerate(attributions):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axs[0].imshow(denormalize_image(inputs[idx], mean=transform.mean, std=transform.std))
        axs[1].imshow(attribution, cmap='Reds')
        [ax.axis('off') for ax in axs]
        plt.tight_layout()
        plt.savefig(f'{savedir}/{batch_size*i + idx}')
        plt.close()
