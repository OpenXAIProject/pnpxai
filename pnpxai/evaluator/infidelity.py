from typing import Optional
import torch
from torch import Tensor
import numpy as np

from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric


class Infidelity(EvaluationMetric):
    def __init__(self, n_perturbations: int = 200, noise_scale: int = 0.2, batch_size: int = 32):
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale
        self.batch_size = batch_size

    @staticmethod
    def _forward_batch(model, inputs, batch_size):
        training_mode = model.training
        model.eval()
        outputs = []
        indices = list(range(len(inputs)))
        while indices:
            curr_indices, indices = indices[:batch_size], indices[batch_size:]
            outputs.append(model(inputs[curr_indices]))
        model.training = training_mode
        return torch.cat(outputs).detach()

    def __call__(
            self,
            model: Model,
            explainer: ExplainerWArgs,
            inputs: torch.Tensor, # sample: Tensor,
            targets: Optional[torch.Tensor] = None, # label: Tensor,
            attributions: Optional[torch.Tensor] = None, # result
        ) -> torch.Tensor:
        device = next(model.parameters()).device
        inputs = inputs.detach()
        outputs = model(inputs)
        n_instances, n_classes = outputs.shape
        if targets is None:
            targets = outputs.argmax(1).tolist()
        preds = (outputs * torch.eye(n_classes)[targets]).sum(dim=-1, keepdim=True)
        if attributions is None:
            attributions = explainer.attribute(inputs, targets, **explainer.kwargs)

        evaluations = []
        for input, target, attr, pred in zip(inputs, targets, attributions, preds):
            # Add Gaussian random noise
            std, mean = torch.std_mean(input)
            perturbed_input = torch.stack([
                input + (
                    self.noise_scale*(torch.randn_like(input)*std+mean)
                ).clamp(max=0).clamp(min=-1)
                for _ in range(self.n_perturbations)
            ])
            perturbed_output = self._forward_batch(model, perturbed_input, self.batch_size)
            perturbed_pred = perturbed_output[:, target]

            pred_diff = pred - perturbed_pred
            dot_prod = torch.mul(perturbed_input, attr).sum(dim=(1,2,3))
            mu = torch.ones_like(dot_prod).to(device)
            scale = torch.mean(
                mu * pred_diff * dot_prod) / torch.mean(mu * dot_prod * dot_prod)
            dot_prod *= scale
            infd = torch.mean(
                mu * torch.square(pred_diff - dot_prod)) / torch.mean(mu)
            evaluations.append(infd)
        return torch.stack(evaluations).detach()
