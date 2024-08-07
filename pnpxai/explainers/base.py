import abc
import sys
from typing import Tuple, Callable, Optional, Union, Type

import copy
from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core._types import ExplanationType

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})


class Explainer(ABC):
    EXPLANATION_TYPE: ExplanationType = "attribution"

    def __init__(
            self,
            model: Module,
            forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
            additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
            **kwargs
    ) -> None:
        self.model = model.eval()
        self.forward_arg_extractor = forward_arg_extractor
        self.additional_forward_arg_extractor = additional_forward_arg_extractor
        self.device = next(model.parameters()).device

    def _extract_forward_args(
            self,
            inputs: Union[Tensor, Tuple[Tensor]]
    ) -> Tuple[Union[Tensor, Tuple[Tensor], Type[None]]]:
        forward_args = self.forward_arg_extractor(inputs) \
            if self.forward_arg_extractor else inputs
        additional_forward_args = self.additional_forward_arg_extractor(inputs) \
            if self.additional_forward_arg_extractor else None
        return forward_args, additional_forward_args

    def copy(self):
        return copy.copy(self)

    def set_kwargs(self, **kwargs):
        clone = self.copy()
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f'{self} has no attribute {k}.')
            setattr(clone, k, v)
        return clone
        
        
    def attribute(
            self,
            inputs: Union[Tensor, Tuple[Tensor]],
            targets: Tensor,
    ) -> Union[Tensor, Tuple[Tensor]]:
        raise NotImplementedError


