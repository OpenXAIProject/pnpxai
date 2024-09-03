import warnings
from typing import Optional

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
        explainer (Optional[Explainer]): The explainer used for evaluation.
        n_iter (Optional[int]): The number of iterations for perturbation.
        epsilon (Optional[float]): The magnitude of random uniform noise.
    """
    def __init__(
        self,
        model: Model,
        explainer: Optional[Explainer] = None,
        n_iter: Optional[int] = 8,
        epsilon: Optional[float] = 0.2,
    ):
        super().__init__(model, explainer)
        self.n_iter = n_iter
        self.epsilon = epsilon
        if explainer is None:
            warnings.warn('[Sensitivity] explainer is not provided. Please set explainer before evaluate.')

    def evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attributions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.
            attributions (Optional[torch.Tensor]): The attributions of the inputs.

        Returns:
            torch.Tensor: The result of the metric evaluation.
        """
        if attributions is None:
            attributions = self.explainer.attribute(inputs, targets)
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
            perturbed_attr = self.explainer.attribute(
                inputs=perturbed.to(self.device),
                targets=target.repeat(self.n_iter),
            )
            # Get maximum of the difference between the perturbed attribution and the original attribution
            attr_norm = torch.linalg.norm(attr).to(self.device)
            attr_diff = attr - perturbed_attr
            sens = max([torch.linalg.norm(diff)/attr_norm for diff in attr_diff])
            evaluations.append(sens)
        return torch.stack(evaluations).to(self.device)


# def pnpxai_sensitivity(
#         model: nn.Module,
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
#         self.explainer.attribute: Callable,
#         attributions: Optional[torch.Tensor],
#         n_iter: int=8,
#         epsilon: float=0.2,
#         **kwargs
#     ) -> torch.Tensor:
#     """
#     Measures the sensitivity of the model's attributions to perturbations.

#     Attributes:
#     """
#     """
#     Computes the sensitivity of attributions.
    
#     Given a `model`, `inputs` and a explainer, the sensitivity is calculated by maximum norm of difference of attributions,
#     ``evalutions = max(norm(explainer(model, inputs) - explainer(model, perturbed)))``.

#     Args:
#         model (Model): The model to evaluate.
#         inputs (torch.Tensor): The input data (N x C x H x W).
#         targets (torch.Tensor): The target labels for the inputs (N x 1).
#         self.explainer.attribute (Callable): The explainer function providing attributions from inputs and targets.
#         attributions (Optional[torch.Tensor]): The attributions of the inputs (default: None).
#         n_iter (int): Number of iterations for perturbation.
#         epsilon (float): Magnitude of random uniform noise.
#         **kwargs: Additional kwargs to compute metric in an experiment. Not required for single usage.
        
#     Reference:
#         C.-K. Yeh, C.-Y. Hsieh, A.S. Suggala, D.I. Inouye, and P. Ravikumar. On the (in)fidelity and sensitivity of attributions. In Proceedings of the NeurIPS (2019).
#     """
#     device = next(model.parameters()).device
#     inputs = inputs.to(device)
#     targets = targets.to(device)
#     if attributions is None:
#         attributions = self.explainer.attribute(inputs, targets)

#     evaluations = []
#     for input, target, attr in zip(inputs, targets, attributions):
#         # Add random uniform noise which ranges [-epsilon, epsilon]
#         perturbed = torch.stack([input]*n_iter)
#         noise = (
#             torch.rand_like(perturbed).to(device) * epsilon * 2 \
#             - epsilon
#         )
#         perturbed += noise
#         # Get perturbed attribution results
#         perturbed_attr = self.explainer.attribute(
#             inputs=perturbed,
#             targets=target.item()
#         )
#         # Get maximum of the difference between the perturbed attribution and the original attribution
#         attr_norm = torch.linalg.norm(attr)
#         attr_diff = attr - perturbed_attr
#         sens = max([torch.linalg.norm(diff)/attr_norm for diff in attr_diff])
#         evaluations.append(sens)
#     return torch.stack(evaluations).detach()
