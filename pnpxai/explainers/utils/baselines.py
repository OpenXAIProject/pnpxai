from typing import Literal, Tuple, Optional, Union, Literal
import copy

import torch
import torchvision.transforms.functional as TF
from optuna.trial import Trial
from pnpxai.core._types import ModalityOrTupleOfModalities, Modality
from pnpxai.evaluator.optimizer.utils import generate_param_key


def token_baseline_function(inputs: torch.Tensor, token_id: int, **kwargs):
    return torch.ones_like(inputs, dtype=torch.long) * token_id

def zero_baseline_function(inputs, **kwargs):
    return torch.zeros_like(inputs)

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

BASELINE_METHODS = {
    **BASELINE_METHODS_FOR_IMAGE,
    **BASELINE_METHODS_FOR_TEXT,
}


BaselineMethod = Literal['zeros', 'invert', 'gaussian_blur', 'mask_token']

class BaselineFunction:
    def __init__(self, method: BaselineMethod, **kwargs):
        self.method = method
        self.kwargs = kwargs

    @property
    def modality(self):
        if self.method in BASELINE_METHODS_FOR_IMAGE:
            return 'image'
        elif self.method in BASELINE_METHODS_FOR_TEXT:
            return 'text'
        else:
            raise KeyError

    @property
    def available_methods(self):
        if self.modality == 'image':
            return BASELINE_METHODS_FOR_IMAGE
        elif self.modality == 'text':
            return BASELINE_METHODS_FOR_TEXT

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


    def suggest_tunables(self, trial: Trial, key: Optional[str]=None):
        method = trial.suggest_categorical(
            generate_param_key(key, 'method'),
            choices=list(self.available_methods.keys()),
        )
        if method in ['zeros', 'invert', 'mask_token']:
            return {'method': method}
        if method == 'gaussian_blur':
            kernel_size_x = trial.suggest_int(
                generate_param_key(key, 'kernel_size_x'),
                low=1, high=11, step=2,
            )
            kernel_size_y = trial.suggest_int(
                generate_param_key(key, 'kernel_size_y'),
                low=1, high=11, step=2,
            )
            sigma_x = trial.suggest_float(
                generate_param_key(key, 'sigma_x'),
                low=.05, high=2., step=.05,
            )
            sigma_y = trial.suggest_float(
                generate_param_key(key, 'sigma_y'),
                low=.05, high=2., step=.05,
            )
            return {
                'method': method,
                'kernel_size_x': kernel_size_x,
                'kernel_size_y': kernel_size_y,
                'sigma_x': sigma_x,
                'sigma_y': sigma_y,
            }
 

def get_default_baseline_function(
    modality: ModalityOrTupleOfModalities,
    mask_token_id: Optional[int]=None
):
    if modality == 'image':
        return BaselineFunction(method='zeros')
    elif modality == 'text':
        return BaselineFunction(method='mask_token', token_id=mask_token_id)
    elif isinstance(modality, tuple):
        return tuple(get_default_baseline_function(m, mask_token_id) for m in modality)
    else:
        raise NotImplementedError(f"There is no default baseline function for '{modality}'.")


BaselineMethodOrFunction = Union[BaselineMethod, BaselineFunction]
