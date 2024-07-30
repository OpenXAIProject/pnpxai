import random
from typing import Union, Optional, Tuple, Sequence
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import TensorDataset

import numba

from pnpxai.utils import reset_model
from pnpxai.core._types import Model
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric
from tsai.all import (
    CrossEntropyLossFlat,
    TSDataLoaders,
    TSStandardize,
    Learner,
    accuracy
)
from fastai.losses import BaseLoss


@numba.njit
def _generate_seq_params(n_items: int, n_feats: int = 1, val_range: int = 100, data=None) -> np.ndarray:
    if data is None:
        data = np.zeros((n_items, n_feats))
        data[:, :2] = np.random.randint(0, val_range, (n_items, 2))

    for i in range(2, n_feats):
        data[:, i] = (data[:, i-1] + data[:, i-2]) % val_range

    return data


@numba.njit
def _mask_offsets(interventions: np.ndarray, offsets: np.ndarray, n_feats: int):
    for idx, offset in enumerate(offsets):
        interventions[idx, offset:] = interventions[idx, :n_feats - offset]
    return interventions


def _generate_offset_interventions(n_items: int, n_feats: int = 1, mask_start_ratio: Tuple[float, float] = None, val_range: int = 100) -> np.ndarray:
    interventions = np.zeros((n_items, n_feats))
    interventions[:2] = 1
    interventions = _generate_seq_params(
        n_items, n_feats, val_range, interventions
    )

    mask_start_ratio = mask_start_ratio if mask_start_ratio is not None else (
        0.3, 1.0)
    offsets = np.random.randint(
        int(mask_start_ratio[0] * n_feats),
        int(mask_start_ratio[1] * n_feats),
        size=n_items
    )
    offsets = n_feats - offsets
    masks = np.expand_dims(np.arange(0, n_feats), 0)
    masks = np.tile(masks, (n_items, 1))
    masks = masks > np.expand_dims(offsets, -1)

    interventions = _mask_offsets(interventions, offsets, n_feats)

    interventions = interventions * masks

    return interventions, masks


class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return ((input - target) ** 2).mean()


class RORTSF(EvaluationMetric):
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
        n_entries: int = 10000,
        mask_start_region: Tuple[float, float] = None,
        n_epochs: int = 5,
    ):
        self.n_entries = n_entries
        self.gen_offset = 3
        self.mask_start_region = mask_start_region if mask_start_region is not None else (
            0.3, 1.0)
        self.n_epochs = n_epochs

    def generate_data(
        self,
        n_entries: int,
        input_shape: list,
        val_range: int = 100,
        noise_range: int = 100,
        seq_axis: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        seq_axis = seq_axis % len(input_shape)
        target_shape = list(input_shape)
        target_shape.append(target_shape.pop(seq_axis))
        *_, dim_ch, dim_seq = target_shape

        params = _generate_seq_params(
            n_entries * dim_ch, dim_seq, val_range
        )
        params = params.reshape(n_entries, dim_ch, dim_seq)

        interventions, masks = _generate_offset_interventions(
            n_entries, dim_seq, self.mask_start_region, noise_range
        )
        interventions, masks = [
            np.tile(np.expand_dims(datum, -2), (1, dim_ch, 1))
            for datum in [interventions, masks]
        ]
        interventions = interventions * val_range / noise_range
        
        params = (params + interventions) % val_range / val_range

        target_shape = list(range(len(input_shape)))
        target_shape.insert(seq_axis, target_shape.pop(-1))

        params = params.transpose(target_shape)
        masks = masks.transpose(target_shape)

        return params, masks

    def split_data(
        self,
        data: np.ndarray,
        split_lens: Sequence[int],
        split_dim: int = -1,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        split_lens = [0, *list(split_lens)]
        return [
            torch.tensor(np.take(
                data,
                range(split_lens[i], split_lens[i+1]),
                axis=split_dim
            ), dtype=dtype)
            for i in range(len(split_lens) - 1)
        ]

    def generate_datasets(
        self,
        data,
        batch_size: Optional[int] = None
    ) -> TSDataLoaders:
        dsets = [TensorDataset(*datum) for datum in data]

        return TSDataLoaders.from_dsets(
            *dsets, bs=batch_size,
            batch_tfms=[TSStandardize()], num_workers=0,
            shuffle_train=True, drop_last=False
        )

    def generate_datasets_from_sample(
        self,
        model: Model,
        inputs: torch.Tensor,
        seq_axis: int = -1,
        with_mask: bool = False
    ):
        device = next(model.parameters()).device

        input_shape = list(inputs.shape)
        in_seq_len = input_shape[seq_axis]
        out_seq_len = model(inputs.to(device)).shape[seq_axis]

        input_shape[seq_axis] = in_seq_len + out_seq_len

        data_train, mask_train = self.generate_data(
            self.n_entries, input_shape, seq_axis=seq_axis)
        data_valid, mask_valid = self.generate_data(
            self.n_entries, input_shape, seq_axis=seq_axis)

        split_lens = (in_seq_len, in_seq_len + out_seq_len)
        x_train, y_train = self.split_data(
            data_train, split_lens, seq_axis, dtype=inputs.dtype
        )
        x_valid, y_valid = self.split_data(
            data_valid, split_lens, seq_axis, dtype=inputs.dtype
        )

        if with_mask:
            (mask_train,), (mask_valid,) = [self.split_data(
                datum, (in_seq_len,), seq_axis, dtype=inputs.dtype
            ) for datum in [mask_train, mask_valid]]
            return self.generate_datasets(
                [(x_train, y_train, mask_train), (x_valid, y_valid, mask_valid)],
                batch_size=input_shape[0]
            )

        return self.generate_datasets(
            [(x_train, y_train), (x_valid, y_valid)],
            batch_size=input_shape[0]
        )

    def train(self, model: Model, inputs: torch.Tensor, seq_axis: int = -1):
        device = next(model.parameters()).device

        dls = self.generate_datasets_from_sample(
            model, inputs, seq_axis
        )
        eval_model = reset_model(model.cpu()).to(device)

        loss_func = BaseLoss(loss_cls=MSELoss, axis=seq_axis, flatten=False)
        learner = Learner(
            dls, eval_model, loss_func=loss_func
        )
        learner.lr_find()
        learner.fit_one_cycle(self.n_epochs, lr_max=1e-3)
        return eval_model

    def evaluate(self, explainer_w_args: ExplainerWArgs, inputs: torch.Tensor, seq_axis: int = -1):
        model = explainer_w_args.explainer.model
        device = next(model.parameters()).device
        dl = self.generate_datasets_from_sample(
            model, inputs, seq_axis, with_mask=True
        ).valid
        score = 0
        evaluated = 0

        for batch in dl:
            x, y, mask = [datum.to(device) for datum in batch]
            attrs = explainer_w_args.attribute(inputs=x, targets=y)
            attrs = attrs - attrs.min(dim=seq_axis, keepdim=True)[0]
            attrs = attrs / attrs.sum(dim=seq_axis, keepdim=True)
            cur_score = (attrs * mask.float()).sum(dim=seq_axis)
            if torch.isnan(cur_score).any():
                continue
            cur_score = cur_score.mean().nan_to_num().item()
            print(cur_score)
            score += cur_score
            evaluated += 1

        return score / max(1, evaluated)

    def eval_ts(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        explainer_w_args: ExplainerWArgs,
        seq_axis: int = -1,
        retrain: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        model = explainer_w_args.explainer.model
        orig_state_dict = model.state_dict()

        if retrain:
            eval_model = self.train(model, inputs, seq_axis)
            model.load_state_dict(eval_model.state_dict())

        score = self.evaluate(explainer_w_args, inputs, seq_axis)
        model.load_state_dict(orig_state_dict)

        return score
