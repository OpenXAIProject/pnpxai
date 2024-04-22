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
    InceptionTime
)
from pnpxai.explainers import TSMule

seed = 2024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_data(dsid: str):
    x_train, y_train, x_test, y_test = get_UCR_data(dsid, return_split=True)
    x, y, splits = combine_split_data([x_train, x_test], [y_train, y_test])
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(x, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(
        dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0
    )
    return dls


dsid = 'ECGFiveDays'
dls = get_data(dsid)

# model = FCN(dls.vars, dls.c)
model = InceptionTime(dls.vars, dls.c)
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


class TransModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.transpose(-1, -2))


for explainer_type in [TSMule]:
    # for explainer_type in [RAP, LRP, IntegratedGradients, Lime, KernelShap]:
    torch.cuda.empty_cache()
    explainer = explainer_type(learn.model)
    params = {"inputs": x, "targets": y}
    attrs = explainer.attribute(**params)
    attrs = normalize(attrs)
    attrs = clear_outliers(attrs)
    visualize(
        x, attrs, os.path.join(cur_dir, f'results/{explainer_type.__name__}.png'), data_idx)
