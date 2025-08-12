from typing import Tuple, Union, Callable, Any, Type, Optional

from torch import Tensor
from torch.nn.modules import Module


TensorOrTupleOfTensors = Union[Tensor, Tuple[Tensor, ...]]
TargetLayer = Union[str, Module]
TargetLayerOrTupleOfTargetLayers = Union[TargetLayer, Tuple[TargetLayer, ...]]


class TunableParameter:
    def __init__(
        self,
        name: str,
        current_value: Any,
        dtype: Type,
        is_leaf: bool,
        space: Optional[Any] = None,
        selector: Optional[Any] = None,
    ):
        self._name = name
        self._current_value = current_value
        self.dtype = dtype
        self.is_leaf = is_leaf
        self._space = space
        self._selector = selector
        if self.is_leaf and self.space is None:
            raise ValueError("If 'is_leaf' is True, 'space' cannot be None.")
        self._disabled = False

    def __repr__(self):
        return repr(self._current_value)

    @property
    def name(self):
        return self._name

    @property
    def current_value(self):
        return self._current_value

    @property
    def space(self):
        return self._space

    @property
    def selector(self):
        return self._selector

    def rename(self, name):
        self._name = name
        return self

    def update_value(self, value):
        self._current_value = value
        return self

    def set_space(self, space: Any):
        self._space = space
        return self

    def set_selector(self, function_selector, set_space=True):
        self._selector = function_selector
        if set_space:
            self.set_space({'choices': function_selector.choices})
        return self

    def is_callable(self):
        return isinstance(self._current_value, Callable)

    @property
    def disabled(self):
        return self._disabled

    def disable(self):
        self._disabled = True

    def enable(self):
        self._disabled = False

    def __call__(self, *args, **kwargs):
        if not self.is_callable():
            raise TypeError(f'{self._current_value} is not callable.')
        return self._current_value(*args, **kwargs)
