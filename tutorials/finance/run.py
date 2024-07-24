import os
import pickle

import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from shap import KernelExplainer
import quantus as qt

from pnpxai.explainers import KernelShap, Lime, LRP, IntegratedGradients
import qt_wrapper as qtw
from model import TabResNet

from shap.plots import waterfall as shap_waterfall_plot
from shap import Explanation

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# Load test data
with open("data/baf/preprocess/X_test.npy", 'rb') as f:
    X_test = np.load(f)

with open("data/baf/preprocess/y_test.npy", 'rb') as f:
    y_test = np.load(f)

test_df = pd.read_csv("data/baf/preprocess/test.csv", index_col=0)

# Load meta data
with open("data/baf/preprocess/metadata.pkl", 'rb') as f:
    meta = pickle.load(f)


model_nn = TabResNet(X_test.shape[1], 2)
model_nn.load_state_dict(torch.load("models/baf/tabresnet.pth"))
model_nn = torch.nn.Sequential(model_nn, torch.nn.Softmax(dim=1))
model_nn.eval()
pass

sample_size = 100
sample_data = X_test[np.random.choice(X_test.shape[0], sample_size, replace=False)]
dataset = torch.utils.data.TensorDataset(torch.tensor(sample_data, dtype=torch.float32))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=sample_data.shape[0], shuffle=False)
X = next(iter(dataloader))[0]
targets = model_nn(X).detach()

shap_values_df = pd.read_csv("explanations/shap_values.csv", index_col=0)
lime_values_df = pd.read_csv("explanations/lime_values.csv", index_col=0)
lrp_values_df = pd.read_csv("explanations/lrp_values.csv", index_col=0)
ig_values_df = pd.read_csv("explanations/ig_values.csv", index_col=0)

def shap_func(model, inputs, targets):
    # Explain models
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    explainer = KernelShap(model)
    baselines = torch.zeros(inputs.shape[1])

    attr = explainer.attribute(
        inputs=inputs,
        targets=targets,
        baselines=baselines,
        n_samples=400,
        feature_mask=torch.arange(X.shape[1]).unsqueeze(0),
    ).numpy()

    return attr

def lime_func(model, inputs, targets):
    # Explain models
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    explainer = Lime(model_nn)
    baselines = torch.zeros(inputs.shape[1])

    attr = explainer.attribute(
        inputs=inputs,
        targets=targets,
        baselines=baselines,
        n_samples=400,
        feature_mask=torch.arange(X.shape[1]).unsqueeze(0),
    ).numpy()

    return attr
def lrp_func(model, inputs, targets):
    # Explain models
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    explainer = LRP(model)

    attr = explainer.attribute(
        inputs=inputs,
        targets=targets,
    ).numpy()

    return attr

def ig_func(model, inputs, targets):
    # Explain models
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    explainer = IntegratedGradients(model)

    attr = explainer.attribute(
        inputs=inputs,
        targets=targets,
    ).numpy()

    return attr

evaluations = {}

explain_funcs = [
    {'name': 'SHAP', 'func': shap_func},
    {'name': 'LIME', 'func': lime_func},
    {'name': 'LRP', 'func': lrp_func},
    {'name': 'IG', 'func': ig_func},
]

# MoRF
metric_name = "MoRF"
evaluations[metric_name] = []

metric = qt.RegionPerturbation(
    patch_size=1, 
    order="morf",
    perturb_baseline=0,
    abs=True,
    normalise=False,
)

for explain_func in explain_funcs:
    score = metric(
        model_nn, 
        x_batch=X.numpy(), 
        y_batch=targets.argmax(dim=1).numpy(), 
        explain_func=explain_func['func']
    )
    evaluations[metric_name].append(
        {
            "explainer": explain_func["name"],
            "value": score,
        }
    )


# LeRF
metric_name = "LeRF"
evaluations[metric_name] = []

metric = qt.RegionPerturbation(
    patch_size=1, 
    order="lerf",
    perturb_baseline=0,
    abs=True,
    normalise=False,
)

for explain_func in explain_funcs:
    score = metric(
        model_nn, 
        x_batch=X.numpy(), 
        y_batch=targets.argmax(dim=1).numpy(), 
        explain_func=explain_func['func']
    )
    evaluations[metric_name].append(
        {
            "explainer": explain_func["name"],
            "value": score,
        }
    )


# Infidelity
metric_name = "Infidelity"
evaluations[metric_name] = []

metric = qtw.Infidelity(
    loss_func="mse", 
    perturb_patch_sizes=[1],
    n_perturb_samples=10,
    perturb_baseline=0,
)

for explain_func in explain_funcs:
    score = metric(
        model_nn, 
        x_batch=X.numpy(), 
        y_batch=targets.argmax(dim=1).numpy(), 
        explain_func=explain_func['func']
    )
    evaluations[metric_name].append(
        {
            "explainer": explain_func["name"],
            "value": score,
        }
    )


# AvgSensitivity
metric_name = "AvgSensitivity"
evaluations[metric_name] = []

metric = qt.AvgSensitivity(
    nr_samples=20,
)

for explain_func in explain_funcs:
    score = metric(
        model_nn, 
        x_batch=X.numpy(), 
        y_batch=targets.argmax(dim=1).numpy(), 
        explain_func=explain_func['func']
    )
    evaluations[metric_name].append(
        {
            "explainer": explain_func["name"],
            "value": score,
        }
    )


# Complexity
metric_name = "Complexity"
evaluations[metric_name] = []

metric = qt.Complexity()

for explain_func in explain_funcs:
    score = metric(
        model_nn, 
        x_batch=X.numpy(), 
        y_batch=targets.argmax(dim=1).numpy(), 
        explain_func=explain_func['func']
    )
    evaluations[metric_name].append(
        {
            "explainer": explain_func["name"],
            "value": score,
        }
    )


# Save evaluations
with open("evaluations.pkl", 'wb') as f:
    pickle.dump(evaluations, f)