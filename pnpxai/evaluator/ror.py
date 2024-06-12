from typing import Union, Optional
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import TensorDataset

from pnpxai.utils import reset_model
from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric
from tsai.all import (
    CrossEntropyLossFlat,
    TSDataLoaders,
    TSStandardize,
    Learner,
    accuracy,
)


class ROR(EvaluationMetric):
    """
    Measures the fidelity of attributions using input perturbations.

    Attributes:
        n_perturbations (int): Number of perturbations to generate.
        noise_scale (int): Scale factor for Gaussian random noise.
        batch_size (int): Batch size for model evaluation.
        grid_size (int): Size of the grid for creating subsets.
        baseline (Union[float, torch.Tensor]): Baseline value for masked subsets.
    """

    def __init__(
        self,
        n_perturbations: int = 150,
        masked_ratio: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 5,
    ):
        self.n_perturbations = n_perturbations
        self.gen_offset = 3
        self.masked_ratio = masked_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def generate_prime_numbers(self, n):
        primes = np.array([2])
        i = 3
        while len(primes) < n:
            if (primes % i).any():
                primes = np.append(primes, i)
            i += 2
        return torch.from_numpy(primes)

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

    def generate_data(self, input_shape: list, n_classes: int) -> torch.Tensor:
        dim_batch, dim_ch, dim_seq = input_shape

        with torch.no_grad():
            # dL
            ids = torch.arange(0, dim_seq) / dim_seq * torch.pi
            base_x = 0
            # base_x = torch.sin(4 * ids) + torch.cos(8 * ids)

            # y.shape = N
            y = torch.arange(0, n_classes).repeat(self.n_perturbations)

            params = self.generate_prime_numbers(
                n_classes).repeat(self.n_perturbations)
            params = params[:, None, None].repeat_interleave(dim_ch, dim=-2)
            spec_x = 4 * params
            # spec_x.shape = (N, C, L)
            spec_x = ids * (spec_x + self.gen_offset)
            phase = torch.randn(self.n_perturbations * n_classes, dim_ch, 1)
            spec_x = torch.pi * phase + spec_x
            # spec_x = (params + 1) * torch.sin(spec_x)
            spec_x = torch.sin(spec_x)

            mask = (torch.rand_like(spec_x) < self.masked_ratio)
            x = torch.where(mask, spec_x, base_x)
            x = x + torch.arange(0, dim_ch)[None, :, None]
            x = x + torch.rand_like(x) / 10

            return x, y, mask
            

    def generate_datasets(self, data) -> TSDataLoaders:
        dsets = [TensorDataset(*datum) for datum in data]

        return TSDataLoaders.from_dsets(
            *dsets, bs=self.batch_size,
            batch_tfms=[TSStandardize()], num_workers=0,
            shuffle_train=True, drop_last=False
        )

    def train(self, model: Model, input_shape: list, n_classes: int):
        device = next(model.parameters()).device
        x_train, y_train, _ = self.generate_data(
            input_shape, n_classes)
        x_valid, y_valid, _ = self.generate_data(
            input_shape, n_classes)

        dls = self.generate_datasets([(x_train, y_train), (x_valid, y_valid)])
        eval_model = reset_model(model.cpu()).to(device)

        learner = Learner(
            dls, eval_model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
        learner.lr_find()
        learner.fit_one_cycle(self.n_epochs, lr_max=1e-3)
        return eval_model

    def evaluate(self, explainer_w_args: ExplainerWArgs, input_shape, n_classes):
        device = explainer_w_args.explainer.device
        data = self.generate_data(input_shape, n_classes)
        dl = self.generate_datasets([data])[0]
        score = 0

        for batch in dl:
            x, y, mask = [datum.to(device) for datum in batch]
            attrs = explainer_w_args.attribute(inputs=x, targets=y)
            attrs = attrs - attrs.min(dim=-1, keepdim=True)[0]
            attrs = attrs / attrs.sum(dim=-1, keepdim=True)
            cur_score = (attrs * mask.float()).sum(dim=-1).mean()
            print(cur_score)
            score += cur_score

        return score / len(dl)

    def eval_ts(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        explainer_w_args: ExplainerWArgs,
        retrain: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        model = explainer_w_args.explainer.model
        orig_state_dict = model.state_dict()
        input_shape = inputs.shape
        n_classes = model(inputs).shape[-1]

        if retrain:
            eval_model = self.train(
                model, input_shape, n_classes
            )
            model.load_state_dict(eval_model.state_dict())

        score = self.evaluate(explainer_w_args, input_shape, n_classes)
        model.load_state_dict(orig_state_dict)

        return score
