from abc import abstractmethod
from typing import Sequence, Dict, Any, Optional, Type

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
        self.reset_runable_metrics()

    @property
    def available_metrics(self) -> Sequence[Type[EvaluationMetric]]:
        return list(map(lambda metric: type(metric), self.metrics))

    @classmethod
    def weigh_metrics(cls, metrics: Dict[str, Any], weights: Dict[str, Any] = None):
        weights = weights or DEFAULT_METRIC_WEIGHTS
        assert sum(weights.values()) == 1, "Sum of weights should be 1."

        weighted_score = 0
        for metric in metrics:
            if metric not in weights or metrics[metric] is None:
                continue
            weighted_score += metrics[metric] * weights[metric]
        return weighted_score

    def is_metric_id_valid(self, metric_id):
        return isinstance(metric_id, int) and 0 <= metric_id < len(self.metrics)

    def reset_runable_metrics(self):
        self._runable_metrics_ids = range(len(self.metrics))

    def set_runable_metrics(self, metrics_ids: Optional[Sequence[int]] = None):
        if metrics_ids is None:
            self.reset_runable_metrics()
            return
        self._runable_metrics_ids = [
            idx for idx in metrics_ids if self.is_metric_id_valid(idx)
        ]

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
        for metric_id in self._runable_metrics_ids:
            if not self.is_metric_id_valid(metric_id):
                continue

            metric = self.metrics[metric_id]
            st = time.time()
            metric_name = class_to_string(metric)
            metrics_scores[metric_name] = metric(
                model, explainer_w_args, inputs, targets, explanations,
            )
            print(f'Computed {metric_name} in {time.time() - st} sec')

        return metrics_scores
