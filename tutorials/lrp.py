import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from zennit.composites import EpsilonPlus
import plotly.graph_objects as go
import plotly.express as px

from pnpxai.explainers import LRP
from pnpxai.explainers.utils.post_process import postprocess_attr

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image


model, transform = get_torchvision_model("vit_b_16")
dataset = get_imagenet_dataset(transform, subset_size=8)
loader = DataLoader(dataset, batch_size=8)
inputs, labels = next(iter(loader))
targets = labels

explainer = LRP(model)
attributions = explainer.attribute(inputs, targets)

attr = attributions[6]
t = .05
lb = attr.quantile(t)
ub = attr.quantile(1-t)
bounded = attr - torch.nn.functional.relu(attr-ub) + torch.nn.functional.relu(-attr+lb)
# bounded = attr
hm = postprocess_attr(bounded)

# # hm = attributions[1].permute(1,2,0).sum(dim=-1)
# # import pdb; pdb.set_trace()

# # img_data = go.Image(z=denormalize_image(inputs[0], mean=transform.mean, std=transform.std))
# # fig = go.Figure(data=img_data)
# # fig.show()

fig = px.imshow(hm)
# fig.add_trace(px.imshow(attributions[0].permute(1,2,0)))
fig.show()
# # print(attributions.shape)

