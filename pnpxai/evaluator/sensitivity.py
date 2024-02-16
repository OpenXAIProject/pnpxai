from typing import Optional
import torch

from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric


class Sensitivity(EvaluationMetric):
    """
    Measures the sensitivity of the model's explanations to perturbations.

    Attributes:
        n_iter (int): Number of iterations for perturbation.
        epsilon (float): Magnitude of random uniform noise.
    """
    def __init__(
            self,
            n_iter: int=8,
            epsilon: float=0.2
        ):
        self.n_iter = n_iter
        self.epsilon = epsilon
        
    def __call__(
            self,
            model: Model,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            explainer_w_args: ExplainerWArgs,
            attributions: Optional[torch.Tensor]=None,
            **kwargs,
        ) -> torch.Tensor:
        """
        Computes the sensitivity of attributions.

        Args:
            model (Model): The model to evaluate.
            explainer_w_args (ExplainerWArgs): The explainer with arguments.
            inputs (torch.Tensor): The input data (N x C x H x W).
            targets (torch.Tensor): The target labels for the inputs (N x 1).
            attributions (Optional[torch.Tensor]): The attributions of the inputs (default: None).
            **kwargs: Additional kwargs to compute metric in an experiment. Not required for single usage.
        """
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)
        if attributions is None:
            attributions = explainer_w_args.attribute(inputs, targets)

        evaluations = []
        for input, target, attr in zip(inputs, targets, attributions):
            # Add random uniform noise which ranges [-epsilon, epsilon]
            perturbed = torch.stack([input]*self.n_iter)
            noise = (
                torch.rand_like(perturbed).to(device) * self.epsilon * 2 \
                - self.epsilon
            )
            perturbed += noise
            # Get perturbed explanation results
            perturbed_attr = explainer_w_args.attribute(
                inputs=perturbed,
                targets=target.item()
            )
            # Get maximum of the difference between the perturbed explanation and the original explanation
            attr_norm = torch.linalg.norm(attr)
            attr_diff = attr - perturbed_attr
            sens = max([torch.linalg.norm(diff)/attr_norm for diff in attr_diff])
            evaluations.append(sens)
        return torch.stack(evaluations).detach()