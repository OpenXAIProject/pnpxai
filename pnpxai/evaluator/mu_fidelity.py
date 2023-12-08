from typing import Optional, Union

import torch
from torchvision import transforms
from scipy.stats import spearmanr

from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric


class MuFidelity(EvaluationMetric):
    def __init__(
            self,
            n_perturbations: int=200,
            noise_scale: int=0.2,
            batch_size: int=32,
            grid_size: int=9,
            baseline: Union[float, torch.Tensor]=0.0,
        ):
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.baseline = baseline

    @property
    def subset_size(self) -> int:
        return int(self.grid_size ** 2 * self.noise_scale)
    
    @staticmethod
    def _forward_batch(model, inputs, batch_size) -> torch.Tensor:
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
            explainer_w_args: ExplainerWArgs,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
            attributions: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, n_classes = outputs.shape
        if targets is None:
            targets = outputs.argmax(1)
        targets = targets.to(device)
        preds = (outputs * torch.eye(n_classes).to(device)[targets]).sum(dim=-1).detach()
        if attributions is None:
            attributions = explainer_w_args.attribute(inputs, targets, **explainer_w_args.kwargs)
        evaluations = []
        for input, target, attr, pred in zip(
            inputs, targets, attributions, preds,
        ):
            repeated = torch.stack([input]*self.n_perturbations)
            # Add Gaussian random noise
            std, mean = torch.std_mean(repeated)
            noise = torch.randn_like(repeated).to(device) * std + mean
            perturbed = self.noise_scale * noise + repeated
            perturbed = torch.minimum(repeated, perturbed)
            perturbed = torch.maximum(repeated-1, perturbed)

            # prepare the random masks that will designate the modified subset (S in original equation)
            subset_mask = torch.randn(
                (self.n_perturbations, self.grid_size ** 2)).to(device)
            subset_mask = torch.argsort(subset_mask, dim=-1) > self.subset_size
            subset_mask = torch.reshape(subset_mask.type(
                torch.float32), (self.n_perturbations, 1, self.grid_size, self.grid_size))
            subset_mask = transforms.Resize(
                perturbed.shape[-2:],
                transforms.InterpolationMode("nearest")
            ).forward(subset_mask)

            # Use the masks to set the selected subsets to baseline state
            masked = perturbed * subset_mask + \
                (1.0 - subset_mask) * self.baseline
            
            masked_output = self._forward_batch(model, masked, self.batch_size)
            pred_diff = pred - masked_output[:, target]

            masked_attr = (attr * (1.0 - subset_mask)).sum(dim=tuple(range(1, subset_mask.ndim)))
            corr, _ = spearmanr(
                pred_diff.cpu().detach().numpy(),
                masked_attr.cpu().detach().numpy(),
            )
            evaluations.append(corr)
        return torch.tensor(evaluations)
