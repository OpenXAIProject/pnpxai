from typing import Sequence, Any, Union, Literal, Optional, Dict

import torch
from torch import Tensor

# from captum.attr import IntegratedGradients as IntegratedGradientsCaptum
from captum._utils.typing import BaselineType, TargetType

from zennit.attribution import IntegratedGradients as IntegratedGradientsZennit

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer


class IntegratedGradients(Explainer):
    """
    Computes Integrated Gradients explanations for a given model.

    Attributes:
    - model (Model): The model for which Integrated Gradients explanations are computed.
    - source (IntegratedGradientsZennit): The Integrated Gradients source for explanations.
    """
    def __init__(self, model: Model):
        """
        Initializes an IntegratedGradients object.

        Args:
        - model (Model): The model for which Integrated Gradients explanations are computed.
        """
        super().__init__(model=model)
        self.source = IntegratedGradientsZennit(self.model)

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        n_classes: Optional[int] = None,
    ) -> Tensor:
        """
        Computes Integrated Gradients attributions for the given inputs.

        Args:
        - inputs (DataSource): The input data.
        - targets (TargetType): The target labels for the inputs (default: None).
        - n_classes (Optional[int]): The number of classes (default: None).

        Returns:
        - Tensor: Integrated Gradients attributions.
        """
        if n_classes is None:
            n_classes = self.model(inputs).shape[-1]
        if isinstance(targets, int):
            targets = [targets] * len(inputs)
        else:
            targets = targets.cpu()
        _, gradients = self.source(
            inputs,
            torch.eye(n_classes)[targets].to(self.device),
        )
        return inputs * gradients

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
