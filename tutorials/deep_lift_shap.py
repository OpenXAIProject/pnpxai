from typing import Callable, Optional
import os
import shap
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pnpxai.explainers import (
    DeepLiftShap,
    Gradient,
    GradientXInput,
    SmoothGrad,
    VarGrad,
    IntegratedGradients,
    LRPUniformEpsilon,
)


#------------------------------------------------------------------------------#
#------------------------------------ data ------------------------------------#
#------------------------------------------------------------------------------#

class PandasDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        transform: Optional[Callable]=None, # e.g. scaler
    ):
        super().__init__()
        self.features = features
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features.iloc[idx]
        label = self.labels.iloc[idx]
        if self.transform is not None:
            features = self.transform(features)
        return features, label


def collate_fn(batch): # to tensor
    inputs = torch.stack([torch.from_numpy(d[0].values) for d in batch]).to(torch.float)
    labels = torch.tensor([d[1] for d in batch]).to(torch.long)
    return inputs, labels


#------------------------------------------------------------------------------#
#----------------------------------- model ------------------------------------#
#------------------------------------------------------------------------------#

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


#------------------------------------------------------------------------------#
#---------------------------------- explain -----------------------------------#
#------------------------------------------------------------------------------#

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# random data
FEATURE_COLUMNS = [
    "gender", "intelligibility", "var_f0_semitones", "var_f0_hz", "avg_energy", 
    "var_energy", "max_energy", "ddk_rate", "ddk_average", "ddk_std", 
    "ddk_pause_rate", "ddk_pause_average", "ddk_pause_std"
]
NROWS = 1000
BATCH_SIZE = 8

random_features = pd.DataFrame(
    data=np.random.randn(NROWS, len(FEATURE_COLUMNS)),
    columns=FEATURE_COLUMNS,
)
random_labels = pd.Series(
    data=np.random.randint(0, 1, NROWS),
)
random_dataset = PandasDataset(
    features=random_features,
    labels=random_labels,
)
dataloader = DataLoader(
    random_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)

# toy model
toy_model = ToyModel(in_features=len(FEATURE_COLUMNS), out_features=2)
toy_model.to(device)
toy_model.eval()


# explainers for mlp and kwargs to construct them
explainers_for_mlp = {
    DeepLiftShap: {'background_data': torch.tensor(random_features.values).to(torch.float).to(device)},
    Gradient: {},
    GradientXInput: {},
    SmoothGrad: {'n_iter': 20, 'noise_level': .1},
    VarGrad: {'n_iter': 20, 'noise_level': .1},
    IntegratedGradients: {'n_steps': 20},
    LRPUniformEpsilon: {'epsilon': 1e-6},
}

# get data to explain
batch = next(iter(dataloader))
features, labels = map(lambda aten: aten.to(device), batch)
targets = toy_model(features).argmax(-1)

# explain
attrs = {}
for explainer_type, explainer_kwargs in explainers_for_mlp.items():
    explainer_nm = explainer_type.__name__
    explainer = explainer_type(model=toy_model, **explainer_kwargs)
    attrs[explainer_nm] = explainer.attribute(inputs=features, targets=targets)
