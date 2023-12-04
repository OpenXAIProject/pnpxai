from typing import Optional
import torch
from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric


class Sensitivity(EvaluationMetric):
    def __init__(self, n_iter: int = 10, epsilon: float = 0.2):
        self.n_iter = n_iter
        self.epsilon = epsilon
        
    def __call__(
            self,
            model: Model,
            explainer: ExplainerWArgs,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor],
            attributions: Optional[torch.Tensor],
        ) -> torch.Tensor:
        device = next(model.parameters()).device
        inputs = inputs.detach()
        if targets is None:
            outputs = model(inputs)
            targets = outputs.argmax(1).tolist()
        if attributions is None:
            attributions = explainer.attribute(inputs, targets, **explainer.kwargs)

        evaluations = []
        for input, target, attr in zip(inputs, targets, attributions):
            perturbed_input = torch.stack([
                input + (
                    torch.rand_like(input).to(device) * self.epsilon * 2 - self.epsilon
                ) for _ in range(self.n_iter)
            ])
            perturbed_attr = explainer.attribute(
                perturbed_input,
                targets = target.item(),
            )
            attr_diff = attr - perturbed_attr
            orig_attr_norm = torch.linalg.norm(attr)
            sens = [torch.linalg.norm(diff)/orig_attr_norm for diff in attr_diff]
            evaluations.append(max(sens))
        return torch.stack(evaluations).detach()