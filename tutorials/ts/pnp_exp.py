import os
import gc
import math
from typing import Sequence, Type

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
        # IntegratedGradients,
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
    metrics = get_metrics([AbPC, Complexity], model, modality, agg_dim)
    metrics = [*metrics, Composite(metrics, composite_agg_func)]
    metrics.extend(get_metrics([Sensitivity], model, modality, agg_dim))
    # metrics = metrics[:2]

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

    log_file = open(
        os.path.join(CURRENT_PATH, f"out/pnp_{model.__class__.__name__}.csv"), "a+"
    )
    plot_path = os.path.join(CURRENT_PATH, "plots", model.__class__.__name__, 'pnp')

    plot_data_step = 100
    plot_data, _ = expr.manager.get_data(data_ids)

    plot_inputs = torch.concat([datum for datum, _ in plot_data], dim=0)
    plot_inputs = tensor_mapper(plot_inputs[::plot_data_step]).to(DEVICE)

    plot_targets = torch.concat([target for _, target in plot_data], dim=0)
    plot_targets = tensor_mapper(plot_targets[::plot_data_step]).to(DEVICE)

    for metric, metric_id in zip(*expr.manager.get_metrics()):
        for explainer, explainer_id in zip(*expr.manager.get_explainers()):
            # try:
            optimized = expr.optimize(
                data_ids=data_ids,
                explainer_id=explainer_id,
                metric_id=metric_id,
                direction=optimization_directions[metric.__class__],
                sampler="tpe",  # Literal['tpe','random']
                n_trials=20,
                seed=42,  # seed for sampler: by default, None
            )

            best_value = optimized.study.best_trial.value / len(data_ids)
            print(
                f"Metric: {metric.__class__.__name__}; Explainer: {explainer.__class__.__name__};"
            )
            print("Best/value:", best_value)  # get the optimized value

            torch.cuda.empty_cache()
            log_data = [
                str(metric.__class__.__name__),
                str(explainer.__class__.__name__),
                str(best_value),
            ]
            log = ",".join(log_data) + "\n"
            print(log)
            log_file.write(log)
            log_file.flush()

            attributions = optimized.explainer.attribute(inputs=plot_inputs, targets=plot_targets)
            attributions = optimized.postprocessor(attributions)
            attributions = attributions.clamp(min=-1 + 1e-9, max=1 - 1e-9).cpu().detach()

            plot_inputs_and_attributions(
                plot_inputs[..., 0, :].tolist(),
                attributions[..., 0, :].tolist(),
                " ".join(log_data),
                os.path.join(
                    plot_path,
                    str(metric.__class__.__name__),
                    str(explainer.__class__.__name__),
                    f"{'_'.join(log_data[:2])}.png",
                ),
            )

            del optimized
            # except:
            #     print(
            #         f"[FAILED!!!] Metric: {metric.__class__.__name__}; Explainer: {explainer.__class__.__name__}"
            #     )
            gc.collect()
            torch.cuda.empty_cache()

    log_file.close()


if __name__ == "__main__":
    app()
