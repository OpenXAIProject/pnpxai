#%%
import os; os.environ['CUDA_VISIBLE_DEVICES']='1'
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
explainer_id = 2 # explainer_id to be optimized (1: Gradient, 2: GradientXInput, 4: IntegratedGradients)
metric_id = 1 # metric_id to be used as objective: AbPC
data_id = 668
explainer_nm = expr.manager.get_explainer_by_id(explainer_id).__class__.__name__
print(explainer_nm)

data = expr.manager.batch_data_by_ids(data_ids=[data_id])
inputs = expr.input_extractor(data)
labels = expr.label_extractor(data)




#%%
# explainer = IntegratedGradients(model, n_steps=70)
# explainer = Gradient(model)
explainer = GradientXInput(model)

modality = ImageModality(channel_dim=1)
default_kwargs = {"feature_mask_fn": modality.get_default_feature_mask_fn(),
                  "baseline_fn": modality.get_default_baseline_fn()}

for k,v in default_kwargs.items():
    if hasattr(explainer, k):
        print(k)
        print(v)
        explainer = explainer.set_kwargs(**{k:v})
        
attrs = explainer.attribute(inputs, labels)

fig, ax = plt.subplots(1,1)
im = attrs[0].detach().cpu().numpy()
im = np.transpose(im, (1,2,0))
im = (im - im.min()) / (im.max() - im.min())
ax.imshow(im)
plt.show()

#%%
diffusion_ckpt_path = '../pnpxai/explainers/diffusion_ckpts/openai-guided-diffusion/256x256_diffusion_uncond.pt'
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

# %%
fig, ax = plt.subplots(1,1)
im = pps[0].detach().cpu().numpy()
ax.imshow(im, cmap='twilight')

# %%
