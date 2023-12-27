from typing import Optional
import torch

from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric


class Sensitivity(EvaluationMetric):
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
            explainer_w_args: ExplainerWArgs,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor]=None,
            attributions: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        if targets is None:
            outputs = model(inputs)
            targets = outputs.argmax(1)
        if attributions is None:
            attributions = explainer_w_args.attribute(inputs, targets, **explainer_w_args.kwargs)

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