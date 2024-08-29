from typing import Literal, Union, Optional, Sequence
import copy
import torch
from skimage.segmentation import (
    felzenszwalb,
    quickshift,
    slic,
    watershed,
)
from optuna.trial import Trial
from pnpxai.explainers.utils.base import UtilFunction
from pnpxai.evaluator.optimizer.utils import generate_param_key


class FeatureMaskFunction(UtilFunction):
    def __init__(self):
        pass

    @staticmethod
    def _skseg_for_tensor(fn, inputs: torch.Tensor, **kwargs):
        feature_mask = [
            torch.tensor(fn(
                inp.permute(1, 2, 0).detach().cpu().numpy(),
                **kwargs
            )) for inp in inputs
        ]
        return torch.stack(feature_mask).long().to(inputs.device)


class Felzenszwalb(FeatureMaskFunction):
    def __init__(
        self,
        scale: float = 250.,
        sigma: float = 1.,
        min_size: int = 50,
    ):
        super().__init__()
        self.scale = scale
        self.sigma = sigma

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            felzenszwalb,
            inputs,
            scale=self.scale,
            sigma=self.sigma,
            min_size=50,
        )

    def get_tunables(self):
        return {
            'scale': (float, {'low': 1e0, 'high': 1e3, 'log': True}),
            'sigma': (float, {'low': 0., 'high': 2., 'step': .05}),
        }


class Quickshift(FeatureMaskFunction):
    def __init__(
        self,
        ratio: float = 1.,
        kernel_size: float = 5,
        max_dist: float = 10.,
        sigma: float = 0.,
    ):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.max_dist = max_dist
        self.sigma = sigma

    def __call__(self, inputs: torch.Tensor):
        if inputs.size(1) == 1:
            inputs = inputs.tile(1, 3, 1, 1)
        return self._skseg_for_tensor(
            quickshift,
            inputs,
            ratio=self.ratio,
            kernel_size=self.kernel_size,
            max_dist=self.max_dist,
            sigma=self.sigma,
        )

    def get_tunables(self):
        return {
            'ratio': (float, {'low': 0., 'high': 1., 'step': .1}),
            'kernel_size': (float, {'low': 1., 'high': 10., 'step': 1.}),
            'max_dist': (int, {'low': 1, 'high': 20, 'step': 1}),
            'sigma': (float, {'low': 0., 'high': 2., 'step': .1,})
        }


class Slic(FeatureMaskFunction):
    def __init__(
        self,
        n_segments: int = 150,
        compactness: float = 1.,
        sigma: float = 0.
    ):
        super().__init__()
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            slic,
            inputs,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
        )

    def get_tunables(self):
        return {
            'n_segments': (float, {'low': 10, 'high': 200, 'step': 10}),
            'compactness': (float, {'low': 1e-2, 'high': 1e2, 'log': True}),
            'sigma': (float, {'low': 0., 'high': 2., 'step': .1}),
        }


class Watershed(FeatureMaskFunction):
    def __init__(
        self,
        markers: int,
        compactness: float,
    ):
        super().__init__()
        self.markers = markers
        self.compactness = compactness

    def __call__(self, inputs: torch.Tensor):
        return self._skseg_for_tensor(
            watershed,
            inputs,
            markers=self.markers,
            compactness=self.compactness,
        )

    def get_tunables(self):
        return {
            'markers': (int, {'low': 10, 'high': 200, 'step': 10}),
            'compactness': (float, {'low': 1e-6, 'high': 1., 'log': True}),
        }


class NoMask1d(FeatureMaskFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        bsz, seq_len = inputs.size()
        seq_masks = torch.arange(seq_len).repeat(bsz).view(bsz, seq_len)
        return seq_masks.to(inputs.device)


FEATURE_MASK_FUNCTIONS_FOR_IMAGE = {
    'felzenszwalb': Felzenszwalb,
    'quickshift': Quickshift,
    'slic': Slic,
    # 'watershed': watershed_for_tensor, TODO: watershed
}

FEATURE_MASK_FUNCTIONS_FOR_TEXT = {
    'no_mask_1d': NoMask1d,
}

FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES = {
    'no_mask_1d': NoMask1d,
}

FEATURE_MASK_FUNCTIONS = {
    **FEATURE_MASK_FUNCTIONS_FOR_IMAGE,
    **FEATURE_MASK_FUNCTIONS_FOR_TEXT,
    **FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES,
}


FeatureMaskMethod = Literal[
    'felzenszwalb', 'quickshift', 'slic', 'watershed', 'no_mask_1d'
]
FeatureMaskMethodOrFunction = Union[FeatureMaskMethod, FeatureMaskFunction]
