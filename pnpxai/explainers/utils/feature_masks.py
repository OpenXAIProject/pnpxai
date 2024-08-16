from typing import Literal, Union, Optional
import copy
import torch
from skimage.segmentation import (
    felzenszwalb,
    quickshift,
    slic,
    watershed,
)
from optuna.trial import Trial
from pnpxai.core._types import ModalityOrTupleOfModalities
from pnpxai.evaluator.optimizer.utils import generate_param_key


def _skseg_for_tensor(fn, inputs: torch.Tensor, **kwargs):
    feature_mask = [
        torch.tensor(fn(
            inp.permute(1,2,0).detach().cpu().numpy(),
            **kwargs
        )) for inp in inputs
    ]
    return torch.stack(feature_mask).long().to(inputs.device)

def felzenszwalb_for_tensor(
    inputs: torch.Tensor,
    scale: float=250.,
    sigma: float=1.,
    **kwargs
):
    return _skseg_for_tensor(
        felzenszwalb,
        inputs,
        scale=scale,
        sigma=sigma,
        min_size=50,
    )

def quickshift_for_tensor(
    inputs: torch.Tensor,
    ratio: float, # [0,1]
    kernel_size: float,
    max_dist: int,
    sigma: float,
    **kwargs
):
    return _skseg_for_tensor(
        quickshift,
        inputs,
        ratio=ratio,
        kernel_size=kernel_size,
        max_dist=max_dist,
        sigma=sigma,
    )

def slic_for_tensor(
    inputs,
    n_segments: int,
    compactness: float, # [0, 100] logscale
    sigma: float,
    **kwargs
):
    return _skseg_for_tensor(
        slic,
        inputs,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
    )

def watershed_for_tensor(
    inputs,
    markers: int,
    compactness: float,
    **kwargs
):
    return _skseg_for_tensor(
        watershed,
        inputs,
        markers=markers,
        compactness=compactness,
    )

def no_mask_for_text(inputs, **kwargs):
    bsz, seq_len = inputs.size()
    seq_masks = torch.arange(seq_len).repeat(bsz).view(bsz, seq_len)
    return seq_masks.to(inputs.device)

FEATURE_MASK_FUNCTIONS_FOR_IMAGE = {
    'felzenszwalb': felzenszwalb_for_tensor,
    'quickshift': quickshift_for_tensor,
    'slic': slic_for_tensor,
    # 'watershed': watershed_for_tensor,
}

FEATURE_MASK_FUNCTIONS_FOR_TEXT = {
    'no_mask': no_mask_for_text,
}

FEATURE_MASK_FUNCTIONS = {
    **FEATURE_MASK_FUNCTIONS_FOR_IMAGE,
    **FEATURE_MASK_FUNCTIONS_FOR_TEXT,
}


class FeatureMaskFunction:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}(method={self.method})"

    @property
    def modality(self):
        if self.method in FEATURE_MASK_FUNCTIONS_FOR_IMAGE:
            return 'image'
        elif self.method in FEATURE_MASK_FUNCTIONS_FOR_TEXT:
            return 'text'
        else:
            raise KeyError

    @property
    def available_methods(self):
        if self.modality == 'image':
            return FEATURE_MASK_FUNCTIONS_FOR_IMAGE
        elif self.modality == 'text':
            return FEATURE_MASK_FUNCTIONS_FOR_TEXT

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
        return FEATURE_MASK_FUNCTIONS[self.method](inputs, **self.kwargs)

    def suggest_tunables(self, trial: Trial, key: Optional[str]=None):
        method = trial.suggest_categorical(
            generate_param_key(key, 'method'),
            choices=self.available_methods.keys(),
        )
        if method == 'felzenszwalb':
            return {
                'method': method,
                'scale': trial.suggest_float(
                  generate_param_key(key, 'scale'),
                    low=1e0, high=1e3, log=True,
                ),
                'sigma': trial.suggest_float(
                  generate_param_key(key, 'sigma'),
                    low=0., high=2., step=.1,
                ),
            }
        elif method == 'quickshift':
            return {
                'method': method,
                'ratio': trial.suggest_float(
                    generate_param_key(key, 'ratio'),
                    low=0., high=1., step=.1,
                ),
                'kernel_size': trial.suggest_float(
                    generate_param_key(key, 'kernel_size'),
                    low=1., high=10., step=.1,
                ),
                'max_dist': trial.suggest_int(
                    generate_param_key(key, 'max_dist'),
                    low=1, high=20, step=1,
                ),
                'sigma': trial.suggest_float(
                    generate_param_key(key, 'sigma'),
                    low=0., high=2., step=.1,
                ),
            }
        elif method == 'slic':
            return {
                'method': method,
                'n_segments': trial.suggest_int(
                    generate_param_key(key, 'n_segments'),
                    low=10, high=200, step=10,
                ),
                'compactness': trial.suggest_float(
                    generate_param_key(key, 'compactness'),
                    low=1e-2, high=1e2, log=True,
                ),
                'sigma': trial.suggest_float(
                    generate_param_key(key, 'sigma'),
                    low=0., high=2., step=.1,
                ),
            }
        elif method == 'watershed':
            return {
                'method': method,
                'markers': trial.suggest_int(
                    generate_param_key(key, 'markers'),
                    low=10, high=200, step=10,
                ),
                'compactness': trial.suggest_float(
                    generate_param_key(key, 'compactness'),
                    low=1e-6, high=1., log=True,
                ),
            }
        elif method == 'no_mask':
            return {'method': method}

FeatureMaskMethod = Literal['felzenszwalb', 'quickshift', 'slic', 'watershed', 'no_mask']
FeatureMaskMethodOrFunction = Union[FeatureMaskMethod, FeatureMaskFunction]

def get_default_feature_mask_fn(modality: ModalityOrTupleOfModalities):
    if modality == 'image':
        return FeatureMaskFunction(method='felzenszwalb', scale=250)
    elif modality == 'text':
        return FeatureMaskFunction(method='no_mask')
    elif isinstance(modality, tuple):
        return tuple(get_default_feature_mask_fn(m) for m in modality)
    else:
        raise NotImplementedError(f"There is no default feature_mask_fn for '{modality}.")