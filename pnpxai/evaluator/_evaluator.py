from dataclasses import dataclass
from collections import OrderedDict
from abc import abstractmethod
from typing import Sequence

from pnpxai.utils import class_to_string
from pnpxai.evaluator._types import EvaluatorOutput
from pnpxai.explainers import Explainer
from pnpxai.core._types import Args

import time

class EvaluatorMetric():
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class XaiEvaluator:
    def __init__(self, metrics: Sequence[EvaluatorMetric]):
        self.metrics = metrics

    def prioritize(self, metrics, weights):
        assert sum(weights) == 1, "Sum of weights should be 1."
        
        weighted_scores = dict()
        weighted_score = 0
        for i, weight in enumerate(weights):
            weighted_score += metrics[i] * weight
        weighted_scores = weighted_score

        weighted_scores = OrderedDict(
            sorted(weighted_scores.items(), key=lambda item: item[1]))
        return weighted_scores

    def __call__(self, sample, label, explainer: Explainer, explanation_outupts):
        model = explainer.model
        pred = model(sample)
        pred_score, pred_idx = pred.topk(1)

        metrics_scores = {}

        # Get attribution score
        for metric in self.metrics:
            st = time.time()
            metric_name = class_to_string(metric)
            metrics_scores[metric_name] = metric(
                model, explainer, sample, label, pred, pred_idx, explanation_outupts
            )
            print(f'Compute {metric_name} done: {time.time() - st}')
            
        return EvaluatorOutput(
            # evaluation_results=weighted_scores,
            metrics_results=metrics_scores,
        )

@dataclass
class EvaluatorWArgs:
    evaluator: XaiEvaluator
    config: Args