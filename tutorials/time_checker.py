from time import time_ns

import torch
from torch.utils.data import DataLoader

import pandas as pd
from pnpxai.detector import ModelArchitectureDetector
from pnpxai.recommender import XaiRecommender
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator.mu_fidelity import MuFidelity
from pnpxai.evaluator.sensitivity import Sensitivity
from pnpxai.evaluator.complexity import Complexity

from helpers import get_imagenet_dataset, get_torchvision_model


def time_recorder(model_name, device="cpu"):
    device = torch.device(device)
    model, transform = get_torchvision_model(model_name)
    model = model.to(device)
    dataset = get_imagenet_dataset(transform=transform, subset_size=8)
    loader = DataLoader(dataset, batch_size=1)
    inputs, labels = next(iter(loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    detector = ModelArchitectureDetector()
    architecture = detector(model=model).architecture
    recommender = XaiRecommender()
    recommended = recommender(question="why", task="image", architecture=architecture)

    records = []
    default_record = {"device": device, "model": model.__class__.__name__}
    for explainer_type in recommended.explainers:
        # TODO: fix RAP for inception
        if model_name == "inception_v3" and explainer_type.__name__ == "RAP":
            continue

        record = {**default_record, "explainer": explainer_type.__name__}
        explainer = ExplainerWArgs(explainer=explainer_type(model))

        started = time_ns()
        attrs = explainer.attribute(inputs, labels)
        record["expl_sec"] = (time_ns() - started)*1e-9

        started = time_ns()
        mufd = MuFidelity()(
            model=model,
            explainer_w_args=explainer,
            inputs=inputs,
            targets=labels,
            attributions=attrs,
        )
        record["mufd_sec"] = (time_ns() - started)*1e-9

        started = time_ns()
        mufd = Sensitivity()(
            model=model,
            explainer_w_args=explainer,
            inputs=inputs,
            targets=labels,
            attributions=attrs,
        )
        record["sens_sec"] = (time_ns() - started)*1e-9

        started = time_ns()
        mufd = Complexity()(
            model=model,
            explainer_w_args=explainer,
            inputs=inputs,
            targets=labels,
            attributions=attrs,
        )
        record["cmpx_sec"] = (time_ns() - started)*1e-9

        records.append(record)
    return records

records = []
for model_name in ["resnet18", "vgg16", "inception_v3", "vit_b_16"]:
    records += time_recorder(model_name)
    if torch.cuda.is_available():
        records += time_recorder(model_name, device="cuda")

pd.DataFrame.from_records(records).to_csv("time_records.csv")

# from pnpxai.explainers import RAP

# model, transform = get_torchvision_model("inception_v3")
# dataset = get_imagenet_dataset(transform=transform, subset_size=8)
# loader = DataLoader(dataset, batch_size=1)
# inputs, labels = next(iter(loader))
# explainer = RAP(model)
# explainer.attribute(inputs, labels)
