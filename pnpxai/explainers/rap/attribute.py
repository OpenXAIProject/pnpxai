from typing import Any, List, Callable

from plotly import express as px
from plotly import graph_objects as go
import numpy as np
import torch
from torch import Tensor, nn
from torch.autograd import Variable

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer
from pnpxai.explainers.rap.rap import RelativeAttributePropagation


class RAP(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = RelativeAttributePropagation(model)
        self.device = next(self.model.parameters()).device

    def compute_pred(self, output):
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        pred = pred.squeeze(-1)

        pred_one_hot = nn.functional.one_hot(pred, 1000) * 1.0
        pred_one_hot = pred_one_hot.to(self.device)
        return pred_one_hot

    def attribute(self, inputs: DataSource, target: DataSource, *args: Any, **kwargs: Any) -> DataSource:
        attributions = []

        datum = inputs.to(self.device)
        outputs = self.model(datum)
        preds = self.compute_pred(outputs)
        relprop = self.method.relprop(preds, *args, **kwargs)
        relprop = relprop.sum(dim=1, keepdim=True)
        attributions = relprop.permute(0, 2, 3, 1)

        return attributions

    def format_outputs_for_visualization(
        self,
        data: DataSource,
        explanations: DataSource,
        input_extractor: Callable[[Any], Any],
        target_extractor: Callable[[Any], Any],
        task: Task,
        kwargs,
    ):
        explanations = explanations.sum(-1)
        return super().format_outputs_for_visualization(
            data=data,
            input_extractor=input_extractor,
            target_extractor=target_extractor,
            explanations=explanations,
            task=task,
            kwargs=kwargs
        )
