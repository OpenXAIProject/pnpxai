from abc import abstractmethod
from typing import Dict, Any
from pnpxai.explainers import ExplainerWArgs
from pnpxai.core._types import Model, DataSource, TensorOrTensorSequence

import time

DEFAULT_METRIC_WEIGHTS = {
    'Sensitivity': 1/3,
    'Complexity': 1/3,
    'MuFidelity': 1/3,
}


class EvaluationMetric():
    @abstractmethod
    def __call__(
        self,
        model: Model,
        explainer_w_args: ExplainerWArgs,
        inputs: DataSource,
        targets: DataSource,
        explanations: TensorOrTensorSequence
    ):
        raise NotImplementedError()


def weigh_metrics(metrics: Dict[str, Any], weights: Dict[str, Any] = None):
    weights = weights or DEFAULT_METRIC_WEIGHTS
    assert sum(weights.values()) == 1, "Sum of weights should be 1."

    weighted_score = 0
    for metric in metrics:
        if metric not in weights or metrics[metric] is None:
            continue
        weighted_score += metrics[metric] * weights[metric]
    return weighted_score