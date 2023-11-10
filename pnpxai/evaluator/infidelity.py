import torch
from torch import Tensor
import numpy as np

from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluatorMetric


class Infidelity(EvaluatorMetric):
    def __init__(self, n_perturbations: int = 200, noise_scale: int = 0.2, batch_size: int = 32):
        self.n_perturbations = n_perturbations
        self.noise_scale = noise_scale
        self.batch_size = batch_size

    def __call__(self, model: Model, explainer_w_args: ExplainerWArgs, sample: Tensor, label, pred, pred_idx, result):
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

        # Compute the dot product of the input perturbation to the explanation
        dot_product = torch.mul(perturbed_sample, result)
        dot_product = dot_product.sum(dim=(1, 2, 3))
        mu = torch.ones(dot_product.shape).to(device)

        def _forward_batch(model, samples, label, batch_size):
            training_mode = model.training
            model.eval()

            predictions = torch.zeros(samples.shape[0]).to(device)
            cur_idx = 0
            next_idx = min(batch_size, samples.shape[0])
            while cur_idx < samples.shape[0]:
                idxs = np.arange(cur_idx, next_idx)
                predictions[idxs] = model(samples[idxs])[:, label].detach()

                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, samples.shape[0])

            model.training = training_mode
            return predictions

        # Compute the difference between the original prediction and the perturbed prediction
        # TODO: Check model(repeated_sample) vs. model(perturbed_sample)
        # perturbed_pred = _forward_batch(model, repeated_sample, label, infidelity_kwargs['batch_size'])
        perturbed_pred = _forward_batch(
            model, perturbed_sample, label, self.batch_size)
        pred_diff = pred - perturbed_pred

        ''' Notes
            - `pred_diff` is extremely small; close to zero.
            - It makes `scaling_factor` to almost zero.
            - `pred_diff - dot_product` is also almost zero.
            - 
            '''

        scaling_factor = torch.mean(
            mu * pred_diff * dot_product) / torch.mean(mu * dot_product * dot_product)
        dot_product *= scaling_factor
        infd = torch.mean(
            mu * torch.square(pred_diff - dot_product)) / torch.mean(mu)

        return infd
