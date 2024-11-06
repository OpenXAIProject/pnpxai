#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pnpxai import AutoExplanationForImageClassification
from pnpxai.explainers.utils.baselines import MeanBaselineFunction

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image

import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pnpxai.explainers import IntegratedGradients, Gradient, GradientXInput, Gfgp
from pnpxai.core.modality.modality import ImageModality
from pnpxai.explainers.utils.postprocess import PostProcessor

#%%
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, transform = get_torchvision_model('resnet18')
dataset = get_imagenet_dataset(transform, indices=range(1000))
loader = DataLoader(dataset, batch_size=4, shuffle=False)

expr = AutoExplanationForImageClassification(
    model=model.to(device),
    data=loader,
    input_extractor=lambda batch: batch[0].to(device),
    label_extractor=lambda batch: batch[-1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device),
    target_labels=False,  # target prediction if False
)


# user inputs
metric_id = 1 # metric_id to be used as objective: AbPC
data_id = 668

data = expr.manager.batch_data_by_ids(data_ids=[data_id])
inputs = expr.input_extractor(data)
labels = expr.label_extractor(data)


diffusion_ckpt_path = None
explainer = Gfgp(model=model, transforms=transform, diffusion_ckpt_path=diffusion_ckpt_path)
modality = ImageModality(channel_dim=1)
default_kwargs = {"feature_mask_fn": modality.get_default_feature_mask_fn(),
                  "baseline_fn": modality.get_default_baseline_fn()}

for k,v in default_kwargs.items():
    if hasattr(explainer, k):
        print(k)
        print(v)
        explainer = explainer.set_kwargs(**{k:v})
        
attrs = explainer.attribute(inputs, labels)

postprocessors = modality.get_default_postprocessors()
postprocessor_id = 0
postprocessor = postprocessors[postprocessor_id]
pps = postprocessor(attrs)

fig, ax = plt.subplots(1,1)
im = pps[0].detach().cpu().numpy()
ax.imshow(im, cmap='twilight')
plt.show()
fig.savefig('gfgp.jpg', bbox_inches='tight', pad_inches=0)
