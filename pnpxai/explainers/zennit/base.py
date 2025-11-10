from typing import Callable, Optional, Union, List, Any

from torch import Tensor
from torch.nn import Module

from pnpxai.explainers.base import Explainer
from pnpxai.utils import format_into_tuple


class ZennitExplainer(Explainer):
    def __init__(
        self,
        model: Module,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        n_classes: Optional[int] = None,
        **kwargs
    ) -> None:
        super().__init__(
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )
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
            formatted = self._wrapped_model.format_inputs(inputs)
            outputs = self._wrapped_model(*formatted)
            self.n_classes = outputs.shape[-1]
        return func(*args, **kwargs)
    return wrapper
