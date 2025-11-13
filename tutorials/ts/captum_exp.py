import os
import math
from typing import Sequence, Type

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import torch
from torch import nn
from torch.utils.data import DataLoader

from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    KernelShap,
    Lime,
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
    tensor_mapper,
    get_composite_agg_func,
    plot_inputs_and_attributions,
    DEVICE
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


def app():
    dsid = "TwoLeadECG"
    loader = get_ts_dataset_loader(dsid, ROOT_PATH, BATCH_SIZE)
    # model = ResNetPlus(loader.vars, loader.c)
    model = PatchTST(loader.vars, None, loader.len, loader.c, classification=True)
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
    inputs, target = (tensor_mapper(datum).to(DEVICE) for datum in data_batch)
    model = model.to(DEVICE)

    explainer_types = [
        InputXGradient,
        IntegratedGradients,
        KernelShap,
        Lime,
    ]
    explainers = get_explainers(explainer_types, model)

    metrics = get_metrics([AbPC, Complexity, Sensitivity], model, modality, agg_dim)
    metrics = [
        Composite([metrics[0], metrics[2]], get_composite_agg_func([0.8, -0.2])), # AbPC, Sensitivity
        Composite(metrics, get_composite_agg_func([0.6, -0.2, -0.2])), # AbPC, Complexity, Sensitivity
    ]

    plot_path = os.path.join(CURRENT_PATH, "plots/composite/", model.__class__.__name__, 'captum')
    os.makedirs(plot_path, exist_ok=True)
    log_path = os.path.join(CURRENT_PATH, f"out/composite/captum_{model.__class__.__name__}.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    log_file = open(log_path, "a+")

    plot_data_step = 100

    for metric in metrics:
        for explainer in explainers:
            try:

                attributions = explainer.attribute(inputs, target=target)
                metric = metric.set_explainer(lambda x, y: explainer.attribute(inputs=x, target=y))
                evals = metric.evaluate(inputs, target, attributions)
                evals = (sum(evals) / len(evals)).item()

                print(
                    f"Metric: {str(metric)}; Explainer: {explainer.__class__.__name__};"
                )
                print("Best/value:", evals)

                torch.cuda.empty_cache()
                log_data = [
                    str(metric),
                    str(explainer.__class__.__name__),
                    str(evals),
                ]
                log = ",".join(log_data) + "\n"
                print(log)
                log_file.write(log)
                log_file.flush()

                attributions = attributions.clamp(min=-1 + 1e-9, max=1 - 1e-9)

                plot_inputs_and_attributions(
                    inputs[::plot_data_step, 0, :].tolist(),
                    attributions[::plot_data_step, 0, :].tolist(),
                    " ".join(log_data),
                    os.path.join(
                        plot_path,
                        str(metric),
                        str(explainer.__class__.__name__),
                        f"{'_'.join(log_data[:2])}.png",
                    ),
                )

            except Exception as e:
                print(
                    f"[FAILED!!!] Metric: {metric.__class__.__name__}; Explainer: {explainer.__class__.__name__}. {e}"
                )

    log_file.close()


if __name__ == "__main__":
    app()
