import torch

from open_xai.core._types import Model
from open_xai.explainers import ExplainerWArgs
from open_xai.evaluator._evaluator import EvaluatorMetric


class Sensitivity(EvaluatorMetric):
    def __init__(self, n_iter: int = 10, epsilon: float = 0.2):
        self.n_iter = n_iter
        self.epsilon = epsilon
        
    def __call__(self, model: Model, explainer_w_args: ExplainerWArgs, sample, label, pred, pred_idx, result):
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
            perturbed_result = explainer_w_args.explainer.attribute(
                *explainer_w_args.args.args,
                inputs=perturbed_sample,
                target=pred_idx,
                **explainer_w_args.args.kwargs
            )

            # Get maximum of the difference between the perturbed explanation and the original explanation
            sens = torch.max(sens, torch.linalg.norm(
                result - perturbed_result)/norm)
        return sens
