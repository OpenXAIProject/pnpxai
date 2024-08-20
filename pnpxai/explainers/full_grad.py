from typing import Optional, Literal

import torchvision.transforms.functional as TF
from torch.nn.modules import Module
from optuna.trial import Trial

from pnpxai.core._types import Tensor
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.zennit.attribution import FullGradient as FullGradAttributor
from pnpxai.evaluator.optimizer.utils import generate_param_key
from pnpxai.utils import format_into_tuple


def _format_pooling_method(method):
    if method == 'abssum':
        return lambda attrs: attrs.abs().sum(1)
    elif method == 'possum':
        return lambda attrs: attrs.clamp(min=0).sum(1)
    else:
        raise ValueError


def _format_interpolate_mode(mode):
    if mode == 'nearest':
        return TF.InterpolationMode.NEAREST
    elif mode == 'nearest_exact':
        return TF.InterpolationMode.NEAREST_EXACT
    elif mode == 'bicubic':
        return TF.InterpolationMode.BICUBIC
    return TF.InterpolationMode.BILINEAR


class FullGrad(ZennitExplainer):
    def __init__(
        self,
        model: Module,
        pooling_method: Literal['abssum', 'possum']='abssum',
        interpolate_mode: Literal['bilinear', 'nearest', 'nearest_exact', 'bicubic']='blinear',
        n_classes: Optional[int]=None,
    ):
        super().__init__(model=model, n_classes=n_classes)
        self.pooling_method = pooling_method
        self.interpolate_mode = interpolate_mode

    def attribute(self, inputs: Tensor, targets: Tensor):
        with FullGradAttributor(
            model=self.model,
            pooling_method=_format_pooling_method(self.pooling_method),
            interpolate_mode=_format_interpolate_mode(self.interpolate_mode),
        ) as attributor:
            attrs = attributor(inputs, self._format_targets(targets))
        return attrs

    def suggest_tunables(self, trial: Trial, key: Optional[str]=None):
        return {
            'pooling_method': trial.suggest_categorical(
                generate_param_key(key, 'pooling_method'),
                choices=['abssum', 'possum'],
            ),
            'interpolate_mode': trial.suggest_categorical(
                generate_param_key(key, 'interpolate_mode'),
                choices=['bilinear', 'nearest', 'nearest_exact', 'bicubic'],
            ),
        }
