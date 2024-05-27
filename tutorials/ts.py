import os
import random

import torch
from torch import nn, Tensor

import numpy as np

from matplotlib import pyplot as plt

from tsai.all import (
    get_UCR_data,
    combine_split_data,
    Categorize,
    TSDatasets,
    TSDataLoaders,
    TSStandardize,
    InceptionTime,
    Learner,
    accuracy,
    PatchTST
)
from pnpxai.explainers import TSMule, RAP
from pnpxai.evaluator import Complexity, MuFidelity, Sensitivity

seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_data(dsid: str, batch_size=32):
    x_train, y_train, x_valid, y_valid = get_UCR_data(dsid, return_split=True, force_download=True)
    x, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(
        dsets.train, dsets.valid, bs=[batch_size, batch_size*2],
        batch_tfms=[TSStandardize()], num_workers=0,
        shuffle_train=False, drop_last=False
    )
    return dls


dsid = 'ECG200'
dls = get_data(dsid)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = FCN(dls.vars, dls.c)
model = InceptionTime(dls.vars, dls.c).to(device)
# model = PatchTST(dls.vars, None, dls.len, dls.c, classification=True).to(device)
learn = Learner(dls, model, metrics=accuracy)
try:
    learn.load('stage0')
except Exception as e:
    learn.save('stage0')
    learn.load('stage0')

try:
    learn.load('stage1')
except Exception as e:
    learn.lr_find()
    learn.fit_one_cycle(25, lr_max=1e-3)
    learn.save('stage1')
    learn.load('stage1')


def threshold_plot(data, confidences, y_min, y_max):
    # Define colors based on confidence RGB
    colors = [
        (1, 1 - c, 1 - c) if c > 0 else (1 + c, 1 + c, 1)
        for c in confidences
    ]

    # Plotting
    plt.plot(data, color='black', label='Lookback')
    for i in range(len(data) - 1):
        plt.fill_betweenx([y_min, y_max],
                          i, i + 1, color=colors[i])


def clear_outliers(x: Tensor, threshold=3, dim=-1) -> Tensor:
    std = x.std(dim=dim, keepdim=True)
    mean = x.mean(dim=dim, keepdim=True)
    low = mean - threshold * std
    high = mean + threshold * std

    x = torch.where(x < low, low, x)
    x = torch.where(x > high, high, x)

    return x


def normalize(x: Tensor, dim=-1) -> Tensor:
    x_max = x.amax(dim=dim, keepdim=True) + 1e-9
    x_min = x.amin(dim=dim, keepdim=True) + 1e-9
    return torch.where(x > 0, x / x_max, -x / x_min)


def visualize(x, attrs, title, idx=0):
    x = x[idx, 0, :]
    attrs = attrs[idx, 0, :]
    threshold_plot(x.tolist(), attrs.tolist(), min(x).item(), max(x).item())
    plt.savefig(title)
    plt.close()


cur_dir = os.path.dirname(os.path.abspath(__file__))

x, y = dls.one_batch()
x = torch.Tensor(x)
y = torch.Tensor(y)

data_idx = 0

metric_types = [MuFidelity, Complexity, Sensitivity]
results = {}

for metric_type in metric_types:
    metric = metric_type()
    for explainer_type in [RAP]:
        # for explainer_type in [RAP, LRP, IntegratedGradients, Lime, KernelShap]:
        torch.cuda.empty_cache()
        explainer = explainer_type(learn.model)
        params = {"inputs": x, "targets": y}
        attrs = explainer.attribute(**params)
        metric_val = metric(
            model=learn.model,
            inputs=x,
            targets=y,
            attributions=attrs
        )

        if explainer_type.__name__ not in results:
            results[explainer_type.__name__] = {}
        results[explainer_type.__name__][metric_type.__name__] = metric_val

        # attrs = normalize(attrs)
        # attrs = clear_outliers(attrs)
        # visualize(
        #     x, attrs, os.path.join(cur_dir, f'results/{explainer_type.__name__}.png'), data_idx)

print(results)