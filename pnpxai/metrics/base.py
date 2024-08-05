import abc
import sys
import warnings
from typing import Optional, Union, Callable

import torch
from torch import nn

from pnpxai.core._types import ExplanationType, Model
from pnpxai.explainers.base import Explainer

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})


class Metric(ABC):
    SUPPORTED_EXPLANATION_TYPE: ExplanationType = "attribution"

    def __init__(
            self,
            model: Model,
            explainer: Optional[Explainer] = None,
            **kwargs
        ):
        self.explainer = explainer
        if isinstance(model, nn.Module):
            self.model = model.eval()
            self.device = next(model.parameters()).device
        else:
            self.model = model

    def set_explainer(self, explainer: Explainer):
        self.explainer = explainer
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