#----------------------------
# ./tutorials/rap_zennit.py
#----------------------------

import torch
import plotly.express as px
from torch.utils.data import DataLoader
from pnpxai.explainers.rap.rap_zennit import RAPZennit
from pnpxai.explainers import RAP
from helpers import get_imagenet_dataset, get_torchvision_model

resnet, transform = get_torchvision_model("vgg16")
dataset = get_imagenet_dataset(transform=transform, indices=range(8))
loader = DataLoader(dataset, batch_size=8)
inputs, labels = next(iter(loader))

explainer = RAPZennit(resnet)
# explainer = RAP(resnet)
attrs = explainer.attribute(inputs, labels)

# postprocess for attributions
def postprocess_attr(attr, sign=None, scale=None):
    if sign == 'absolute':
        attr = torch.abs(attr)
    elif sign == 'positive':
        attr = torch.nn.functional.relu(attr)
    elif sign == 'negative':
        attr = -torch.nn.functional.relu(-attr)

    postprocessed = attr.permute((1, 2, 0)).sum(dim=-1)
    attr_max = torch.max(postprocessed)
    attr_min = torch.min(postprocessed)
    postprocessed = (postprocessed - attr_min) / (attr_max - attr_min)
    if scale == "sqrt":
        postprocessed = postprocessed.sqrt()
    return postprocessed.cpu().detach().numpy()

fig = px.imshow(postprocess_attr(attrs[0]), color_continuous_scale="RdBu_r")
fig.show()