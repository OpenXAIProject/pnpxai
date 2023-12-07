import torch
from torch import Tensor
from torchvision import transforms
from scipy.stats import spearmanr
import numpy as np

from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric


class MuFidelity(EvaluationMetric):
    def __init__(self, n_perturbations: int = 200, noise_scale: int = 0.2, batch_size: int = 32, grid_size: int = 9, baseline: float or Tensor = 0.0):
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.baseline = baseline

    def __call__(self, model: Model, explainer_w_args: ExplainerWArgs, sample: Tensor, label, pred, pred_idx, result):
        init_pred_shape = pred.shape
        pred = pred[:, label]
        print(self.n_perturbations)
        repeated_sample = sample.repeat(
            self.n_perturbations, *([1]*(sample.ndim - 1))
        )

        device = next(model.parameters()).device

        # Add Gaussian random noise
        std, mean = torch.std_mean(repeated_sample)
        noise = torch.randn(size=repeated_sample.shape).to(device) * std + mean
        perturbed_sample = self.noise_scale * noise + repeated_sample
        perturbed_sample = torch.minimum(repeated_sample, perturbed_sample)
        perturbed_sample = torch.maximum(repeated_sample-1, perturbed_sample)

        # Generate subset masks
        sample_size = perturbed_sample.size()
        # cardinal of subset (|S| in the equation)
        subset_size = int(self.grid_size ** 2 * self.noise_scale)
        # prepare the random masks that will designate the modified subset (S in original equation)
        subset_mask = torch.randn(
            (sample_size[0], self.grid_size ** 2)).to(device)
        subset_mask = torch.argsort(subset_mask, dim=-1) > subset_size
        subset_mask = torch.reshape(subset_mask.type(
            torch.float32), (sample_size[0], 1, self.grid_size, self.grid_size))
        subset_mask = transforms.Resize(
            perturbed_sample.shape[-2:], transforms.InterpolationMode("nearest")).forward(subset_mask)

        # Use the masks to set the selected subsets to baseline state
        masked_sample = perturbed_sample * subset_mask + \
            (1.0 - subset_mask) * self.baseline

        def _forward_batch(model, samples, label, batch_size):
            training_mode = model.training
            model.eval()
            n_samples = samples.shape[0]

            predictions = torch.zeros(
                n_samples, *list(init_pred_shape)[1:]
            ).to(device)
            cur_idx = 0
            next_idx = min(batch_size, n_samples)
            while cur_idx < n_samples:
                idxs = np.arange(cur_idx, next_idx)
                predictions[idxs] = model(samples[idxs])[:, label].detach()

                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, samples.shape[0])

            model.training = training_mode
            return predictions

        # Compute the difference between the original prediction and the masked prediction
        masked_pred = _forward_batch(
            model, masked_sample, label, self.batch_size)
        pred_diff = pred - masked_pred

        attrs = result * (1.0 - subset_mask)
        attrs = attrs.sum(dim=(-3, -2, -1))
        corr, p = spearmanr(
            masked_pred.cpu().detach().numpy(),
            attrs.cpu().detach().numpy(),
        )

        return corr
