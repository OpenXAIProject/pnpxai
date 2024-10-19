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
explainer_id = 2 # explainer_id to be optimized
metric_id = 1 # metric_id to be used as objective: AbPC
data_id = 668
explainer_nm = expr.manager.get_explainer_by_id(explainer_id).__class__.__name__
print(explainer_nm)

#%%
# optimize
optimized = expr.optimize(
    data_ids=data_id,
    explainer_id=explainer_id,
    metric_id=metric_id,
    direction='maximize',
    sampler='random',
    n_trials=50,
    seed=42,
)

# prepare trials to visualize
sorted_trials = sorted(
    [trial for trial in optimized.study.trials if trial.value is not None],
    key=lambda trial: trial.value,
)
trials_to_vis = {
    'Worst': sorted_trials[0],
    'Median': sorted_trials[len(sorted_trials)//2],
    'Best': sorted_trials[-1],
}

data = expr.manager.batch_data_by_ids(data_ids=[data_id])
inputs = expr.input_extractor(data)
labels = expr.label_extractor(data)


#%%
sorted_trials[-1].__dict__
#%%
sorted_trials[-1].__dict__['_params']
#%%
sorted_trials[-1].__dict__['_user_attrs']

#%%
sorted_trials[-1].user_attrs

#%%
title='Best'
trial = trials_to_vis[title]
explainer = trial.user_attrs['explainer']
postprocessor = trial.user_attrs['postprocessor']

attrs = explainer.attribute(inputs, labels)
pps = postprocessor(attrs)

fig, ax = plt.subplots(1,1, figsize=(5,5))
im = inputs.detach().cpu().numpy()[0]
im = np.transpose(im, (1,2,0))
ax.imshow(im)

fig, ax = plt.subplots(1,1)
ax.imshow(pps[0].detach().cpu().numpy(), cmap='seismic')

fig, ax = plt.subplots(1,1)
im = attrs[0].detach().cpu().numpy()
im = np.transpose(im, (1,2,0))
im = (im - im.min()) / (im.max() - im.min())
ax.imshow(im)

#%%
type(postprocessor)

#%%
from pnpxai.explainers import IntegratedGradients, Gradient, GradientXInput
from pnpxai.core.modality.modality import ImageModality
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
# %%
explainer.get_tunables()
#%%
integratedgradients = IntegratedGradients(model)
print(integratedgradients.get_tunables())
gradient = Gradient(model)
print(gradient.get_tunables())
gradientxinput = GradientXInput(model)
print(gradientxinput.get_tunables())
