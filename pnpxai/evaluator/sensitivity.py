import torch

from pnpxai.core._types import Model
from pnpxai.explainers import Explainer
from pnpxai.evaluator._evaluator import EvaluatorMetric


class Sensitivity(EvaluatorMetric):
    def __init__(self, n_iter: int = 10, epsilon: float = 0.2):
        self.n_iter = n_iter
        self.epsilon = epsilon
        
    def __call__(self, model: Model, explainer: Explainer, sample, label, pred, pred_idx, result):
        norm = torch.linalg.norm(result)

        device = next(model.parameters()).device

        sens = torch.tensor(-torch.inf)
        epsilon = self.epsilon
        for _ in range(self.n_iter):
            # Add random uniform noise which ranges [-epsilon, epsilon]
            noise = torch.rand(size=sample.shape).to(device)
            noise = noise * epsilon * 2 - epsilon
            perturbed_sample = sample + noise

            # Get perturbed explanation results
            perturbed_result = explainer.attribute(
                inputs = perturbed_sample,
                targets = pred_idx,
            )

            # Get maximum of the difference between the perturbed explanation and the original explanation
            sens = torch.max(sens, torch.linalg.norm(
                result - perturbed_result)/norm)
        return sens
