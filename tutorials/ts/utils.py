import os

import torch

from matplotlib import pyplot as plt

from tsai.all import (
    get_UCR_data,
    combine_split_data,
    Categorize,
    TSDatasets,
    TSDataLoaders,
    TSStandardize,
    Learner,
    accuracy,
    TSTensor,
)

# ------------------------------------------------------------------------------#
# -------------------------------- basic usage ---------------------------------#
# ------------------------------------------------------------------------------#

# setup
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')
torch.set_num_threads(1)
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(CURRENT_PATH, "../")
BATCH_SIZE = 64


def composite_agg_func(metric1, metric2):
    device = metric1.device
    return metric1 * 0.8 - 0.2 * metric2.to(device)


def get_ts_dataset_loader(dataset: str, path: str, batch_size: int = 64):
    x_train, y_train, x_valid, y_valid = get_UCR_data(dataset, path, return_split=True)
    x, y, splits = combine_split_data([x_train, x_valid], [y_train, y_valid])
    dsets = TSDatasets(x, y, tfms=[None, [Categorize()]], splits=splits, inplace=True)
    return TSDataLoaders.from_dsets(
        dsets.train, dsets.valid, bs=batch_size, batch_tfms=[TSStandardize()]
    )


def train_model(
    loader: TSDataLoaders,
    model: torch.nn.Module,
    model_path: str,
    epochs: int = 15,
    lr: float = 1e-3,
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    learner = Learner(loader, model, metrics=accuracy)
    try:
        learner.load(model_path)
    except:
        learner.lr_find()
        plt.close()
        learner.fit_one_cycle(epochs, lr_max=lr)
        learner.save(model_path)
        torch.cuda.empty_cache()

    return learner.model


def tensor_mapper(x: TSTensor):
    return torch.from_numpy(x.cpu().numpy())
