from typing import Literal, Optional, Union, Literal, Sequence
import copy

import torch
import torchvision.transforms.functional as TF
from optuna.trial import Trial
from pnpxai.evaluator.optimizer.utils import generate_param_key


def token_baseline_function(inputs: torch.Tensor, token_id: int, **kwargs):
    return torch.ones_like(inputs, dtype=torch.long) * token_id


def zero_baseline_function(inputs, **kwargs):
    return torch.zeros_like(inputs)


def mean_baseline_function(inputs, target_dim: int = -1, **kwargs):
    return torch.mean(inputs, dim=target_dim, keepdim=True)


def invert_baseline_function(inputs, **kwargs):
    return TF.invert(inputs)


def gaussian_blur_baseline_function(
    inputs,
    kernel_size_x,
    kernel_size_y,
    sigma_x,
    sigma_y,
    **kwargs
):
    return TF.gaussian_blur(
        inputs,
        kernel_size=[kernel_size_x, kernel_size_y],
        sigma=[sigma_x, sigma_y],
    )


BASELINE_METHODS_FOR_IMAGE = {
    'zeros': zero_baseline_function,
    'invert': invert_baseline_function,
    'gaussian_blur': gaussian_blur_baseline_function,
}

BASELINE_METHODS_FOR_TEXT = {
    'mask_token': token_baseline_function,
}

BASELINE_METHODS_FOR_TIME_SERIES = {
    'zeros': zero_baseline_function,
    'mean': mean_baseline_function,
}

BASELINE_METHODS = {
    **BASELINE_METHODS_FOR_IMAGE,
    **BASELINE_METHODS_FOR_TEXT,
    **BASELINE_METHODS_FOR_TIME_SERIES,
}


BaselineMethod = Literal['zeros', 'invert', 'gaussian_blur', 'mask_token']


class BaselineFunction:
    def __init__(self, method: BaselineMethod, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}(method={self.method})"

    def copy(self):
        return copy.copy(self)

    def set_kwargs(self, **kwargs):
        clone = self.copy()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(clone, k, v)
            else:
                clone.kwargs[k] = v
        return clone

    def __call__(self, inputs: torch.Tensor):
        return BASELINE_METHODS[self.method](inputs, **self.kwargs)

    def suggest_tunables(self, trial: Trial, key: Optional[str] = None):
        choice_sets = [BASELINE_METHODS_FOR_IMAGE, BASELINE_METHODS_FOR_TEXT, BASELINE_METHODS_FOR_TIME_SERIES]
        choices = sum([
            list(choice_set.keys())
            for choice_set in choice_sets
            if self.method in choice_set
        ], [])

        return suggest_tunable_baselines(choices, trial, key)


def suggest_tunable_baselines(baselines: Sequence[str], trial: Trial, key: str) -> dict:
    method = trial.suggest_categorical(
        generate_param_key(key, 'method'),
        choices=baselines,
    )
    return _suggest_tunable_baseline_params(method, trial, key)


def _suggest_tunable_baseline_params(method: str, trial: Trial, key: Optional[str] = None):
    return {
        'zeros': {'method': method},
        'invert': {'method': method},
        'mask_token': {'method': method},
        'gaussian_blur': {
            'method': method,
            'kernel_size_x': trial.suggest_int(
                generate_param_key(key, 'kernel_size_x'),
                low=1, high=11, step=2,
            ),
            'kernel_size_y': trial.suggest_int(
                generate_param_key(key, 'kernel_size_y'),
                low=1, high=11, step=2,
            ),
            'sigma_x': trial.suggest_float(
                generate_param_key(key, 'sigma_x'),
                low=.05, high=2., step=.05,
            ),
            'sigma_y': trial.suggest_float(
                generate_param_key(key, 'sigma_y'),
                low=.05, high=2., step=.05,
            ),
        }
    }.get(method, {})


BaselineMethodOrFunction = Union[BaselineMethod, BaselineFunction]
