from dataclasses import dataclass
from collections import OrderedDict
from abc import abstractmethod
from typing import Sequence

from open_xai.utils import class_to_string
from open_xai.evaluator._types import EvaluatorOutput
from open_xai.explainers import ExplainerWArgs
from open_xai.core._types import Args

import time

# @dataclass
# class EvaluatorOutput:
#     explainers: list
#     evaluation_results: list

class EvaluatorMetric():
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class XaiEvaluator:
    def __init__(self, metrics: Sequence[EvaluatorMetric]):
        self.metrics = metrics
        pass

    def prioritize(self, metrics, weights):
        assert sum(weights) == 1, "Sum of weights should be 1."
        
        weighted_scores = dict()
        for method in metrics[0].keys():
            weighted_score = 0
            for i, weight in enumerate(weights):
                weighted_score += metrics[i][method] * weight
            weighted_scores[method] = weighted_score

        weighted_scores = OrderedDict(
            sorted(weighted_scores.items(), key=lambda item: item[1]))
        return weighted_scores

    def __call__(self, sample, label, explainer_w_args: ExplainerWArgs, explanation_outupts):
        self.device = sample.device
        model = explainer_w_args.explainer.model
        pred = model(sample)
        pred_score, pred_idx = pred.topk(1)

        metrics_scores = {}

        method = class_to_string(explainer_w_args.explainer)

        # Get attribution score
        for metric in self.metrics:
            st = time.time()
            metric_name = class_to_string(metric)
            metrics_scores[metric_name][method] = metric(
                model, explainer_w_args, sample, label, pred, pred_idx, explanation_outupts
            )
            print(f'Compute {metric_name} done: {time.time() - st}')

        # Prioritize explanation results by weighted score
        n_metrics = len(metrics_scores)
        weighted_scores = self.prioritize(
            metrics=metrics_scores.values(),
            weights=[1 / len(n_metrics)] * n_metrics
        )

        # return EvaluatorOutput(
        #     explainers=...,
        #     evaluation_results=...,
        # )
        return EvaluatorOutput(
            evaluation_results=weighted_scores,
            metrics_results=metrics_scores,
        )

@dataclass
class EvaluatorWArgs:
    evaluator: XaiEvaluator
    config: Args