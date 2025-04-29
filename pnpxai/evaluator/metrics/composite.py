from copy import deepcopy
from typing import Sequence, Optional

import torch
from pnpxai.evaluator.metrics import Metric
from pnpxai.explainers.base import Explainer


class Composite(Metric):
    def __init__(self, metrics: Sequence[Metric], agg_func: callable):
        self.metrics = metrics
        self.agg_func = agg_func

    def set_explainer(self, explainer: Explainer) -> "Composite":
        clone = self.copy()
        clone.metrics = [metric.set_explainer(explainer) for metric in clone.metrics]
        return clone

    def evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attributions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.agg_func(
            *[
                metric.evaluate(
                    inputs=inputs, targets=targets, attributions=attributions, **kwargs
                )
                for metric in self.metrics
            ]
        )
