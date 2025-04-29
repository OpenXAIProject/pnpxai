import os
from typing import Sequence, Type

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

import shap
from pnpxai.evaluator.metrics import AbPC, Complexity, Composite, Metric, Sensitivity
from pnpxai.core.modality.modality import TimeSeriesModality

from tsai.all import (
    PatchTST,
    ResNetPlus,
)


from tutorials.ts.utils import (
    get_ts_dataset_loader,
    train_model,
    CURRENT_PATH,
    ROOT_PATH,
    BATCH_SIZE,
    DEVICE,
    tensor_mapper,
    composite_agg_func,
)


def get_explainers(explainer_types, model):
    return [exp_type(model) for exp_type in explainer_types]


def get_metrics(
    metric_types: Sequence[Type[Metric]],
    model: nn.Module,
    modality: TimeSeriesModality,
    agg_dim: int,
):
    default_kwargs = {
        "baseline_fn": modality.get_default_baseline_fn(),
        "feature_mask_fn": modality.get_default_feature_mask_fn(),
        "channel_dim": modality.channel_dim,
        "mask_agg_dim": agg_dim,
    }

    metrics = []
    for metric_type in metric_types:
        metric = metric_type(model=model)
        for k, v in default_kwargs.items():
            if hasattr(metric, k):
                metric = metric.set_kwargs(**{k: v})
        metrics.append(metric)

    return metrics


def build_shap_attribute(explainer: shap.KernelExplainer):
    def wrapper(inputs: torch.Tensor, targets: torch.Tensor):
        device = inputs.device
        in_shape = inputs.shape
        b_size = in_shape[0]
        if torch.is_tensor(inputs):
            inputs = inputs.detach().cpu().numpy()
        if inputs.ndim > 2:
            inputs = np.reshape(inputs, (b_size, -1))
        attributions = explainer.shap_values(inputs)
        attributions = torch.from_numpy(attributions).reshape(list(in_shape) + [-1])
        target_ids = [...] + [None] * (attributions.ndim - targets.ndim)
        attributions = torch.take_along_dim(
            attributions, targets[target_ids].cpu(), -1
        )[..., 0]
        return attributions.to(device)

    return wrapper


def build_model_wrapper(model, target_shape):
    def wrapper(inputs):
        inputs = np.reshape(inputs, target_shape)
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        inputs = inputs.to(DEVICE)
        return model(inputs).cpu().detach().numpy()

    return wrapper


def app():
    dsid = "TwoLeadECG"
    loader = get_ts_dataset_loader(dsid, ROOT_PATH, BATCH_SIZE)
    # model = ResNetPlus(loader.vars, loader.c)
    model = PatchTST(loader.vars, None, loader.len, loader.c, classification=True)
    model = model.to(DEVICE)
    model = train_model(
        loader,
        model,
        f"{ROOT_PATH}/data/models/{str(model.__class__.__name__).lower()}/{dsid}",
        epochs=40,
    )

    seq_dim = -1
    agg_dim = -2
    modality = TimeSeriesModality(seq_dim)

    test_data = DataLoader(loader.valid.dataset, batch_size=len(loader.valid.dataset))
    data_batch = next(iter(test_data))
    inputs, target = (tensor_mapper(datum) for datum in data_batch)
    baseline = torch.zeros_like(inputs)[:1].reshape(1, -1)
    target_shape = [-1, *list(inputs.shape)[1:]]

    explainer = shap.KernelExplainer(
        model=build_model_wrapper(model, target_shape),
        data=baseline.numpy(),
        link="identity",
    )
    explainer.attribute = build_shap_attribute(explainer)

    metrics = []
    metrics = get_metrics([AbPC, Complexity], model, modality, agg_dim)
    metrics: Sequence[Metric] = [*metrics, Composite(metrics, composite_agg_func)]
    # metrics.extend(get_metrics([Sensitivity], model, modality, agg_dim))
    metrics = metrics[2:]

    log_file = open(
        os.path.join(CURRENT_PATH, f"out/omni_{model.__class__.__name__}.csv"), "a+"
    )

    for metric in metrics:
        attributions = explainer.attribute(inputs, target)
        # print(attributions.shape)
        # evals = metric.evaluate(
        #     inputs.to(DEVICE), target.to(DEVICE), attributions.to(DEVICE)
        # )

        metric.explainer = explainer.attribute
        evals = metric.evaluate(inputs.to(DEVICE), target.to(DEVICE), attributions)

        evals = (sum(evals) / len(evals)).item()

        print(
            f"Metric: {metric.__class__.__name__}; Explainer: {explainer.__class__.__name__};"
        )
        print("Best/value:", evals)

        torch.cuda.empty_cache()
        log_file.write(
            f"{metric.__class__.__name__},{explainer.__class__.__name__},{evals}\n"
        )
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    app()
