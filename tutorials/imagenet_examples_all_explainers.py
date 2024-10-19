#%%
from helpers import get_torchvision_model
model, transform = get_torchvision_model("resnet18")

from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from helpers import get_imagenet_dataset

# load data
indices = [0, 7, 14, 403]
dataset = get_imagenet_dataset(
    transform=transform,
    indices=indices,
    root_dir="./data/ImageNet"
)
dataloader = DataLoader(dataset=dataset, batch_size=len(indices))
inputs, labels = next(iter(dataloader))

#%%
# denormalize input data
def denormalize_image(inputs, mean, std):
    return (
        inputs
        * torch.Tensor(std)[:, None, None]
        + torch.Tensor(mean)[:, None, None]
    ).permute(1, 2, 0).detach().numpy()

# show images
nrows, ncols = 1, 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))

for c, (input, label) in enumerate(zip(inputs, labels)):
    axes[c].imshow(denormalize_image(input, mean=transform.mean, std=transform.std))
    axes[c].set_title(dataset.dataset.idx_to_label(label.item()))
    axes[c].set_xticks([])
    axes[c].set_yticks([])

#%%
from pnpxai.explainers import AVAILABLE_EXPLAINERS

print([explainer.__name__ for explainer in AVAILABLE_EXPLAINERS])

#%%
from pnpxai.explainers import Explainer

explainer = Explainer(model)
attributions = explainer.attribute(inputs, targets, **kwargs)

#%%
from pnpxai.explainers import LRPBase

explainer = LRPBase(model)
attrs_lrp = explainer.attribute(inputs=inputs, targets=labels)
#%%
attrs_lrp_custom = explainer.attribute(inputs=inputs, targets=labels, epsilon=.7)

#%%
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

# visualize attributions
nrows, ncols = 4, 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
for r, (input, attr_default, attr_custom) in enumerate(zip(inputs, attrs_lrp, attrs_lrp_custom)):
    axes[r, 0].imshow(denormalize_image(input, mean=transform.mean, std=transform.std))
    axes[r, 1].imshow(postprocess_attr(attr_default, sign="absolute", scale="sqrt"), cmap="Reds")
    axes[r, 2].imshow(postprocess_attr(attr_custom, sign="absolute", scale="sqrt"), cmap="Reds")
    if r == 0:
        axes[r, 0].set_title("original image")
        axes[r, 1].set_title("lrp: e=0.25 (default)")
        axes[r, 2].set_title("lrp: e=0.70 (custom)")

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    
#%%
from pnpxai.explainers import IntegratedGradients

ig = IntegratedGradients(model)
attrs_ig = ig.attribute(inputs=inputs, targets=labels)

#%%
from pnpxai.explainers import Lime

lime = Lime(model)
attrs_lime = lime.attribute(inputs=inputs, targets=labels)

#%%
from pnpxai.explainers import KernelShap

ks = KernelShap(model)
attrs_ks = ks.attribute(inputs=inputs, targets=labels)
#%%
from pnpxai.explainers import GradCam

gcam = GradCam(model)
attrs_gcam = gcam.attribute(inputs=inputs, targets=labels)
#%%
from pnpxai.explainers import GuidedGradCam

ggcam = GuidedGradCam(model)
attrs_ggcam = ggcam.attribute(inputs=inputs, targets=labels)
#%%
visdata_all = [
    inputs,
    attrs_lrp,
    attrs_ig,
    attrs_lime,
    attrs_ks,
    attrs_gcam,
    attrs_ggcam,
]

nrows, ncols = len(inputs), len(visdata_all)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
for r, visdata in enumerate(zip(*visdata_all)):
    axes[r, 0].imshow(denormalize_image(visdata[0], mean=transform.mean, std=transform.std))
    axes[r, 1].imshow(postprocess_attr(visdata[1], sign="absolute", scale="sqrt"), cmap="Reds")
    axes[r, 2].imshow(postprocess_attr(visdata[2], sign="absolute", scale="sqrt"), cmap="Reds")
    axes[r, 3].imshow(postprocess_attr(visdata[3], sign="absolute", scale="sqrt"), cmap="Reds")
    axes[r, 4].imshow(postprocess_attr(visdata[4], sign="absolute", scale="sqrt"), cmap="Reds")
    axes[r, 5].imshow(denormalize_image(visdata[0], mean=transform.mean, std=transform.std))
    axes[r, 5].imshow(postprocess_attr(visdata[5], sign="absolute"), cmap="jet", alpha=.5)
    axes[r, 6].imshow(denormalize_image(visdata[0], mean=transform.mean, std=transform.std))
    axes[r, 6].imshow(postprocess_attr(visdata[6], sign="absolute"), cmap="jet", alpha=.5)

for r, (input, attr_default, attr_custom) in enumerate(zip(inputs, attrs_lrp, attrs_lrp_custom)):
    if r == 0:
        axes[r, 0].set_title("original image")
        axes[r, 1].set_title("LRP")
        axes[r, 2].set_title("IntegratedGradients")
        axes[r, 3].set_title("Lime")
        axes[r, 4].set_title("KernelShap")
        axes[r, 5].set_title("GradCam")
        axes[r, 6].set_title("GuidedGradCam")

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])