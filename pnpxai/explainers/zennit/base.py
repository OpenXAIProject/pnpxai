from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from ..base import Explainer


class ZennitExplainer(Explainer):
    def __init__(
        self,
        model: Module,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        n_classes: Optional[int]=None,
        **kwargs
    ) -> None:
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.n_classes = n_classes


    def __init_subclass__(cls) -> None:
        cls.attribute = set_n_classes_before(cls.attribute)
        return super().__init_subclass__()

    def _format_targets(self, targets):
        return torch.eye(self.n_classes)[targets.tolist()].to(self.device)


def set_n_classes_before(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.n_classes is None:
            inputs = kwargs.get("inputs") or args[1]
            if isinstance(inputs, Tensor):
                inputs = (inputs,)
            if self.n_classes is None:
                outputs = self.model(*inputs)
                self.n_classes = outputs.shape[-1]
        return func(*args, **kwargs)
    return wrapper