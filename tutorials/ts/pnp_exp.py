import os
import gc
import math
from typing import Sequence, Type

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from pnpxai import Experiment
from pnpxai.explainers import (
    Gradient,
    GradientXInput,
    IntegratedGradients,
    KernelShap,
    Lime,
    LRPEpsilonAlpha2Beta1,
    LRPEpsilonGammaBox,
    LRPEpsilonPlus,
    LRPUniformEpsilon,
    SmoothGrad,
    VarGrad,
    Explainer,
)
from pnpxai.evaluator.metrics import AbPC, Complexity, Composite, Metric, Sensitivity
from pnpxai.core.modality.modality import TimeSeriesModality

import optuna

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
    get_composite_agg_func,
    plot_inputs_and_attributions,
)


def get_explainers(
    explainer_types: Sequence[Type[Explainer]],
    model: nn.Module,
    modality: TimeSeriesModality,
):
    default_kwargs = {
        "feature_mask_fn": modality.get_default_feature_mask_fn(),
        "baseline_fn": modality.get_default_baseline_fn(),
    }
    explainers = []
    for explainer_type in explainer_types:
        explainer = explainer_type(model=model)
        for k, v in default_kwargs.items():
            if hasattr(explainer, k):
                explainer = explainer.set_kwargs(**{k: v})
        explainers.append(explainer)

    return explainers


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


def app():
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    dsid = "TwoLeadECG"
    loader = get_ts_dataset_loader(dsid, ROOT_PATH, BATCH_SIZE)
    # model = ResNetPlus(loader.vars, loader.c)
    model = PatchTST(loader.vars, None, loader.len, loader.c, classification=True)
    model = train_model(
        loader,
        model.to(DEVICE),
        f"{ROOT_PATH}/data/models/{str(model.__class__.__name__).lower()}/{dsid}",
        epochs=40,
    )

    print(
        f"Train shape: {len(loader.train.dataset)}; Test shape: {len(loader.valid.dataset)}"
    )

    test_data = DataLoader(loader.valid.dataset, batch_size=BATCH_SIZE, shuffle=False)
    data_ids = list(range(len(test_data.dataset)))
    # data_ids = list(range(BATCH_SIZE))

    seq_dim = -1
    agg_dim = -2
    modality = TimeSeriesModality(seq_dim)

    explainer_types = [
        Gradient,
        GradientXInput,
        IntegratedGradients,
        KernelShap,
        Lime,
        LRPEpsilonAlpha2Beta1,
        LRPEpsilonGammaBox,
        LRPEpsilonPlus,
        LRPUniformEpsilon,
        SmoothGrad,
        VarGrad,
    ]
    explainers = get_explainers(explainer_types, model, modality)

    metrics = []
    metrics = get_metrics([AbPC, Complexity, Sensitivity], model, modality, agg_dim)
    metrics = [
        Composite(
            [metrics[0], metrics[2]], get_composite_agg_func([0.8, -0.2])
        ),  # AbPC, Sensitivity
        Composite(
            metrics, get_composite_agg_func([0.6, -0.2, -0.2])
        ),  # AbPC, Complexity, Sensitivity
    ]

    expr = Experiment(
        model=model.to(DEVICE),
        data=test_data,
        modality=modality,
        explainers=explainers,
        postprocessors=modality.get_default_postprocessors(),
        metrics=metrics,
        input_extractor=lambda batch: tensor_mapper(batch[0]).to(DEVICE),
        label_extractor=lambda batch: tensor_mapper(batch[-1]).to(DEVICE),
        target_extractor=lambda outputs: outputs.argmax(-1).to(DEVICE),
        target_labels=False,
    )

    expr.predict_batch(data_ids)

    optimization_directions = {
        Complexity: "minimize",
        AbPC: "maximize",
        Composite: "maximize",
        Sensitivity: "minimize",
    }

    plot_path = os.path.join(
        CURRENT_PATH, "plots/composite/", model.__class__.__name__, "pnp"
    )
    os.makedirs(plot_path, exist_ok=True)
    log_path = os.path.join(
        CURRENT_PATH, f"out/composite/pnp_{model.__class__.__name__}.csv"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_file = open(log_path, "a+")

    plot_data_step = 100

    for metric, metric_id in zip(*expr.manager.get_metrics()):
        for explainer, explainer_id in zip(*expr.manager.get_explainers()):
            best_value = 0

            plot_inputs = []
            plot_attrs = []

            for idx, data_id in enumerate(data_ids):
                try:
                    optimized = expr.optimize(
                        data_ids=[data_id],
                        explainer_id=explainer_id,
                        metric_id=metric_id,
                        direction=optimization_directions[metric.__class__],
                        sampler="tpe",  # Literal['tpe','random']
                        n_trials=20,
                        seed=42,  # seed for sampler: by default, None
                    )
                    cur_best_value = optimized.study.best_trial.value
                    best_value += cur_best_value
                    print(
                        f"[{idx + 1}] {str(metric)} | {explainer.__class__.__name__} = {cur_best_value}"
                    )  # get the optimized value

                    if idx % plot_data_step != 0:
                        continue

                    plot_datum, _ = expr.manager.get_data([data_id])
                    plot_datum = next(iter(plot_datum))
                    plot_in, plot_tgt = [
                        tensor_mapper(plot_tensor).to(DEVICE)
                        for plot_tensor in plot_datum
                    ]

                    plot_attr = optimized.explainer.attribute(
                        inputs=plot_in, targets=plot_tgt
                    )
                    plot_attr = optimized.postprocessor(plot_attr)
                    plot_attr = plot_attr.clamp(min=-1 + 1e-9, max=1 - 1e-9)

                    plot_inputs.append(plot_in.cpu().detach().numpy())
                    plot_attrs.append(plot_attr.cpu().detach().numpy())

                    del optimized
                except Exception as e:
                    print(
                        f"[FAILED!!!] Metric: {metric.__class__.__name__}; Explainer: {explainer.__class__.__name__} with error:\n{e}"
                    )
                gc.collect()
                torch.cuda.empty_cache()

            best_value /= len(data_ids)
            print(f"Metric: {str(metric)}; Explainer: {explainer.__class__.__name__};")
            print("Best/value:", best_value)  # get the optimized value

            torch.cuda.empty_cache()
            log_data = [
                str(metric),
                str(explainer.__class__.__name__),
                str(best_value),
            ]
            log = ",".join(log_data) + "\n"
            print(log)
            log_file.write(log)
            log_file.flush()

            plot_inputs = np.concatenate(plot_inputs)
            plot_attrs = np.concatenate(plot_attrs)

            plot_inputs_and_attributions(
                plot_inputs[..., 0, :].tolist(),
                plot_attrs[..., 0, :].tolist(),
                " ".join(log_data),
                os.path.join(
                    plot_path,
                    str(metric),
                    str(explainer.__class__.__name__),
                    f"{'_'.join(log_data[:2])}.png",
                ),
            )

            gc.collect()
            torch.cuda.empty_cache()

    log_file.close()


if __name__ == "__main__":
    app()
