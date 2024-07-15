import abc
import sys
import warnings
from typing import Optional, Union, Callable

import torch
from torch import nn

from pnpxai.core._types import ExplanationType
from pnpxai.explainers.base import Explainer

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})


class Metric(ABC):
    SUPPORTED_EXPLANATION_TYPE: ExplanationType = "attribution"

    def __init__(
            self,
            model: nn.Module,
            explainer: Optional[Explainer] = None,
            **kwargs
        ):
        self.model = model.eval()
        self.explainer = explainer
        self.device = next(model.parameters()).device

    def evaluate(
            self,
            inputs: Union[torch.Tensor, None],
            targets: Union[torch.Tensor, None],
            attributions: Union[torch.Tensor, None],
            **kwargs
        ) -> torch.Tensor:
        """Main function of the interpreter."""
        raise NotImplementedError