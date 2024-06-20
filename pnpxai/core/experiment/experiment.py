from typing import Any, Optional, Callable, Dict, Sequence

import csv
import uuid
import os
import json
import warnings
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from torch.nn.modules import Module

from pnpxai.core.recommender import XaiRecommender
from pnpxai.explainers.base import Explainer
from pnpxai.metrics.base import Metric
from .utils import camel_to_snake, _format_to_tuple
from .file_utils import mkdir, find, exists
from .._types import Modality, Task, Model, DataSource


ROOT_DIR = "./pnpxai-runs"
INPUT_DIR = "inputs"
ATTRIBUTION_DIR = "attrs"
EXPLAINER_CONFS_FILENAME = "explainers.json"
METRIC_CONFS_FILENAME = "metrics.json"
ANNOTATION_FILENAME = "annotations.csv"
ANNOTATION_FIELD_BASE = [
    "input_idx",
    "input_path",
    "label",
    "pred",
    "prob_score",
    "explainer_nm",
    "attr_path",
]
EVALUATION_FIELDS = ["metric_nm", "value"]


class ExperimentRecords:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

        self.explainer_confs = self._load_explainer_confs()
        self.metric_confs = self._load_metric_confs()
        self.annotations = self._load_annotations()
    
    @property
    def experiment_dir(self):
        return os.path.join(ROOT_DIR, self.experiment_name)
    
    @property
    def input_dir(self):
        dir = os.path.join(self.experiment_dir, INPUT_DIR)
        if os.path.exists(dir):
            return dir
        return None
    
    @property
    def attr_dir(self):
        return os.path.join(self.experiment_dir, ATTRIBUTION_DIR)
    
    @property
    def explainer_confs_path(self):
        return os.path.join(self.experiment_dir, EXPLAINER_CONFS_FILENAME)

    @property
    def metric_confs_path(self):
        path = os.path.join(self.experiment_dir, METRIC_CONFS_FILENAME)
        if os.path.exists(path):
            return path
        return None

    @property
    def record_path(self):
        return os.path.join(self.experiment_dir, ANNOTATION_FILENAME)
    
    def _load_explainer_confs(self):
        with open(self.explainer_confs_path, "r") as f:
            explainer_confs = json.load(f)
        return explainer_confs

    def _load_metric_confs(self):
        if self.metric_confs_path is not None:
            with open(self.metric_confs_path, "r") as f:
                metric_confs = json.load(f)
            return metric_confs
        return None

    def _load_annotations(self):
        with open(self.record_path, "r") as f:
            dataiter = csv.DictReader(f)
            annotations = [r for r in dataiter]
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annon = self.annotations[idx]

        # format or load values
        record = {}
        for k, _v in annon.items():
            if k.endswith("_path"):
                k = f"{k.split('_')[0]}s"
                try:
                    v = tuple(torch.tensor(ary) for ary in np.load(_v).values())
                except:
                    import pdb; pdb.set_trace()
                if len(v) == 1:
                    v = v[0]
            elif _v.replace(".","").isnumeric():
                v = eval(_v)
            else:
                v = _v
            record[k] = v
        return record

    


def run_experiment(
    name: str,
    modality: Modality,
    task: Task,
    model: Model,
    dataloader: DataSource,
    input_extractor: Optional[Callable]=lambda data: data[0],
    label_extractor: Optional[Callable]=lambda data: data[-1],
    
    target_fn: Optional[Callable]=lambda outputs: outputs.argmax(1),
    prob_fn: Optional[Callable]=lambda outputs: outputs.softmax(1).max(1).values,
    explainers: Optional[Dict[str, Explainer]]=None,
    metrics: Optional[Dict[str, Metric]]=None,
    recommend: bool=False,
    save_inputs: bool=True,
    target_labels: bool=False, # If True, target_fn is ignored
) -> ExperimentRecords:
    if recommend:
        recommender = XaiRecommender()
        recommended = recommender.recommend(modality=modality, model=model)
        if explainers is None:
            explainers = {
                camel_to_snake(explainer_type.__name__): explainer_type(model)
                for explainer_type in recommended.explainers
            }
        if metrics is None:
            metrics = {
                camel_to_snake(metric_type.__name__): metric_type(model)
                for metric_type in recommended.metrics
            }
    assert explainers is not None, (
        "Must have explainers. If you want to run experiment with explainers "
        "recommended by XaiRecommender, please set recommend=True."
    )
    record_fields = ANNOTATION_FIELD_BASE
    if metrics is None:
        warnings.warn(
            "Evaluation results will not provided because metrics are "
            "not given. If you want to run experiment with metrics "
            "recommended by XaiRecommender, please set recommend=True."
        )
    else:
        record_fields += EVALUATION_FIELDS

    # create dirs and files
    expr_dir, input_dir, attr_dir, record_filepath = _create_experiment_dir(
        name, save_inputs=save_inputs,
    )

    # create record file
    with open(record_filepath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(record_fields)

    # create explainer conf file
    with open(os.path.join(expr_dir, EXPLAINER_CONFS_FILENAME), "w") as f:
        explainer_confs = _extract_confs(explainers)
        json.dump(explainer_confs, f, indent=4)

    # create metric conf file
    if metrics is not None:
        metric_confs = _extract_confs(metrics)
        with open(os.path.join(expr_dir, METRIC_CONFS_FILENAME), "w") as f:
            json.dump(metric_confs, f, indent=4)

    max_digit = len(str(len(dataloader.dataset)))
    for batch_count, data in enumerate(tqdm(dataloader, total=len(dataloader))):
        inputs = _format_to_tuple(input_extractor(data))
        input_idxs = range(
            batch_count*dataloader.batch_size,
            (batch_count+1)*dataloader.batch_size
        )

        # save inputs
        input_paths = []
        for input_idx, *input in zip(input_idxs, *inputs):
            if save_inputs:
                input_path = os.path.join(
                    input_dir,
                    f"input{str(input_idx).zfill(max_digit)}.npz"
                )
                np.savez(input_path, *(inp.cpu().detach().numpy() for inp in input))
                input_paths.append(input_path)
            else:
                input_paths.append(None)

        labels = label_extractor(data)
        outputs = model(*inputs)
        probs = prob_fn(outputs)
        targets = labels if target_labels else target_fn(outputs)
        base_record_values = [
            input_idxs,
            input_paths,
            labels,
            targets,
            probs,
        ]
        for explainer_nm, explainer in explainers.items():
            # explain
            attrs = _format_to_tuple(explainer.attribute(inputs, targets))

            # save explanations
            attr_paths = []
            for input_idx, *attr in zip(input_idxs, *attrs):
                attr_path = os.path.join(
                    attr_dir,
                    f"{explainer_nm}_input{str(input_idx).zfill(max_digit)}.npz"
                )
                np.savez(attr_path, *(a.cpu().detach().numpy() for a in attr))
                attr_paths.append(attr_path)

            # add saved path to record
            explainer_record_values = base_record_values.copy()
            explainer_record_values.append(attr_paths)

            # write records
            if metrics is None:
                rows = [[
                     # input_idx, input_path, label, target
                    r[0], r[1], r[2].item(), r[3].item(),
                     # prob, explainer_nm, attr_path
                    r[4].item(), explainer_nm, r[5],
                ] for r in zip(*explainer_record_values)]
                with open(record_filepath, "a") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                continue

            for metric_nm, metric in metrics.items():
                # evaluate
                evs = metric.evaluate(inputs, targets, attrs, explainer.attribute)

                # add evaluations to record
                metric_record_values = explainer_record_values.copy()
                metric_record_values.append(evs)

                # write records
                rows = [[
                    r[0], r[1].item(), r[2].item(), r[3].item(),
                    explainer_nm, r[4], metric_nm, r[5].item(),
                ] for r in zip(*metric_record_values)]
                with open(record_filepath, "a") as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
    return ExperimentRecords(experiment_name=name)


def _create_experiment_dir(name, save_inputs=True):
    if name is None:
        name = str(uuid.uuid4())
    if not exists(ROOT_DIR):
        mkdir(ROOT_DIR)
    if not find(ROOT_DIR, name):
        mkdir(ROOT_DIR, name) # expr dir
        if save_inputs:
            mkdir(os.path.join(ROOT_DIR, name), INPUT_DIR) # input dir
        mkdir(os.path.join(ROOT_DIR, name), ATTRIBUTION_DIR) # expl dir
    expr_dir = os.path.join(ROOT_DIR, name)
    input_dir = os.path.join(expr_dir, INPUT_DIR) if save_inputs else None
    attr_dir = os.path.join(expr_dir, ATTRIBUTION_DIR)
    record_filepath = os.path.join(expr_dir, ANNOTATION_FILENAME)
    return expr_dir, input_dir, attr_dir, record_filepath


def _format_conf_value(v: Any):
    if isinstance(v, Module):
        return f"{v.__module__}.{v.__class__.__name__}"
    elif isinstance(v, str|bool|float|int):
        return v
    elif isinstance(v, Sequence):
        return [_format_conf_value(elem) for elem in v]
    elif isinstance(v, Callable):
        return f"{v.__module__}.{v.__name__}"
    else:
        return str(v)


def _extract_confs(explainers_or_metrics) -> Dict[str, Dict[str, Any]]:
    return {
        obj_nm: {
            "type": f"{obj.__module__}.{obj.__class__.__name__}",
            "kwargs": {
                k: _format_conf_value(v) for k, v in obj.__dict__.items()
                if (
                    k not in ["model", "device"]
                    and not k.startswith("_")
                )
            },
        } for obj_nm, obj in explainers_or_metrics.items()
    }
