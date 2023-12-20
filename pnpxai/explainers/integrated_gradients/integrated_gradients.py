from typing import Sequence, Any, Union, Literal, Optional, Dict

import torch
from torch import Tensor

# from captum.attr import IntegratedGradients as IntegratedGradientsCaptum
from captum._utils.typing import BaselineType, TargetType

from zennit.attribution import IntegratedGradients as IntegratedGradientsZennit

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer


class IntegratedGradients(Explainer):
    def __init__(self, model: Model):
        super().__init__(model=model)
        self.source = IntegratedGradientsZennit(self.model)

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        n_classes: Optional[int] = None,
    ) -> Tensor:
        if n_classes is None:
            n_classes = self.model(inputs).shape[-1]
        if isinstance(targets, int):
            targets = [targets] * len(inputs)
        else:
            targets = targets.cpu()
        _, attributions = self.source(
            inputs,
            torch.eye(n_classes)[targets].to(self.device),
        )
        return inputs * attributions

    def format_outputs_for_visualization(
        self,
        inputs: DataSource,
        targets: DataSource,
        explanations: DataSource,
        task: Task,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if kwargs.get("return_convergence_delta", False):
            explanations = explanations[0]
        explanations = explanations.transpose(-1, -3)\
            .transpose(-2, -3)
        return super().format_outputs_for_visualization(
            inputs=inputs,
            targets=targets,
            explanations=explanations,
            task=task,
            kwargs=kwargs
        )
