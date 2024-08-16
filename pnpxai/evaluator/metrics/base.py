import abc
import sys
import warnings
from typing import Optional, Union, Callable, Tuple

import copy
import torch
from torch import nn

from pnpxai.core._types import ExplanationType
from pnpxai.explainers import GradCam
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.postprocess import PostProcessor

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})

NON_DISPLAYED_ATTRS = [
    'model',
    'explainer',
    'device',
    'prob_fn',
    'pred_fn',
]

class Metric(ABC):
    SUPPORTED_EXPLANATION_TYPE: ExplanationType = "attribution"

    def __init__(
        self,
        model: nn.Module,
        explainer: Optional[Explainer]=None,
        **kwargs
    ):
        self.model = model.eval()
        self.explainer = explainer
        self.device = next(model.parameters()).device

    def __repr__(self):
        displayed_attrs = ', '.join([
            f'{k}={v}' for k, v in self.__dict__.items()
            if k not in NON_DISPLAYED_ATTRS and v is not None]
        )
        return f"{self.__class__.__name__}({displayed_attrs})"

    def copy(self):
        return copy.copy(self)

    def set_explainer(self, explainer: Explainer):
        assert self.model is explainer.model, 'Must have same model of metric.'
        clone = self.copy()
        clone.explainer = explainer
        return clone

    def set_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def evaluate(
            self,
            inputs: Union[torch.Tensor, None],
            targets: Union[torch.Tensor, None],
            attributions: Union[torch.Tensor, None],
            **kwargs
        ) -> torch.Tensor:
        """Main function of the interpreter."""
        raise NotImplementedError