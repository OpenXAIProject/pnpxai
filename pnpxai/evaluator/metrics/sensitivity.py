import warnings
from typing import Optional, Union, Callable

import torch

from pnpxai.core._types import Model
from pnpxai.explainers.base import Explainer
from pnpxai.evaluator.metrics.base import Metric


class Sensitivity(Metric):
    """
    Computes the complexity of attributions.
    
    Given `attributions`, calculates a fractional contribution distribution `prob_mass`,
    ``prob_mass[i] = hist[i] / sum(hist)``. where ``hist[i] = histogram(attributions[i])``.

    The complexity is defined by the entropy,
    ``evaluation = -sum(hist * ln(hist))``
    
    
    Args:
        model (Model): The model used for evaluation
        explainer (Optional[Union[Explainer, Callable]]): The explainer used for evaluation. It can be an instance of ``Explainer`` or any callable returning attributions from inputs and targets.
        n_iter (Optional[int]): The number of iterations for perturbation.
        epsilon (Optional[float]): The magnitude of random uniform noise.
    """
    def __init__(
        self,
        model: Model,
        explainer: Optional[Union[Explainer, Callable]] = None,
        n_iter: Optional[int] = 8,
        epsilon: Optional[float] = 0.2,
    ):
        super().__init__(model, explainer)
        self.n_iter = n_iter
        self.epsilon = epsilon

    def evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attributions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.
            attributions (Optional[torch.Tensor]): The attributions of the inputs.

        Returns:
            torch.Tensor: The result of the metric evaluation.
        """
        if self.explainer is None:
            warnings.warn('[Sensitivity] explainer is not provided. Please set explainer before evaluate.')
        explain_func = self.explainer.attribute if isinstance(self.explainer, Explainer) else self.explainer
        if attributions is None:
            attributions = explain_func(inputs, targets)
            # attributions = self.explainer.attribute(inputs, targets)
        attributions = attributions.to(self.device)
        evaluations = []
        for inp, target, attr in zip(inputs, targets, attributions):
            # Add random uniform noise which ranges [-epsilon, epsilon]
            perturbed = torch.stack([inp]*self.n_iter)
            noise = (
                torch.rand_like(perturbed).to(self.device) * self.epsilon * 2 \
                - self.epsilon
            )
            perturbed += noise
            # Get perturbed attribution results
            perturbed_attr = explain_func(
                perturbed.to(self.device),
                target.repeat(self.n_iter),
            )
            # Get maximum of the difference between the perturbed attribution and the original attribution
            attr_norm = torch.linalg.norm(attr).to(self.device)
            attr_diff = attr.to(self.device) - perturbed_attr.to(self.device)
            sens = max([torch.linalg.norm(diff)/attr_norm for diff in attr_diff])
            evaluations.append(sens)
        return torch.stack(evaluations).to(self.device)

