from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

from pnpxai.explainers.base import Explainer
from pnpxai.utils import format_into_tuple
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


def set_n_classes_before(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        inputs = kwargs.get('inputs', None)
        if inputs is None:
            inputs = args[1]
        if isinstance(inputs, Tensor):
            inputs = format_into_tuple(inputs)
        if self.n_classes is None:
            outputs = self.model(*inputs)
            self.n_classes = outputs.shape[-1]
        return func(*args, **kwargs)
    return wrapper