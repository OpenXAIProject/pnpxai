from collections import OrderedDict
from abc import abstractmethod
from typing import Sequence, Dict, Any

from torch import Tensor

from pnpxai.utils import class_to_string
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


class XaiEvaluator:
    def __init__(self, metrics: Sequence[EvaluationMetric]):
        self.metrics = metrics

    @classmethod
    def weigh_metrics(cls, metrics: Dict[str, Any], weights: Dict[str, Any] = None):
        weights = weights or DEFAULT_METRIC_WEIGHTS
        assert sum(weights.values()) == 1, "Sum of weights should be 1."

        weighted_score = 0
        for metric in metrics:
            weighted_score += metrics[metric] * weights[metric]
        return weighted_score

    def __call__(
        self,
        inputs: Tensor,
        targets: Tensor,
        explainer_w_args: ExplainerWArgs,
        explanations: Tensor
    ) -> Dict[str, Tensor]:
        model = explainer_w_args.explainer.model
        metrics_scores = {}

        # Get attribution score
        for metric in self.metrics:
            st = time.time()
            metric_name = class_to_string(metric)
            metrics_scores[metric_name] = metric(
                model, explainer_w_args, inputs, targets, explanations,
            )
            print(f'Computed {metric_name} in {time.time() - st} sec')

        return metrics_scores
