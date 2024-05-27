import random
from io import TextIOWrapper
from contextlib import contextmanager
from typing import Sequence, Callable, Any, Union, Optional

import numpy as np
import torch
from torch import Tensor, nn


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_to_string(object):
    return str(object.__class__.__name__)


class CustomIterator():
    def __init__(self, data: Sequence, mapper: Callable[[Any], Any]):
        self.data = data
        self.mapper = mapper
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.data)
        return self

    def __next__(self):
        datum = next(self.iterator)
        return self.mapper(datum)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Observable:
    def __init__(self):
        self._callbacks = []

    def subscribe(self, callback):
        self._callbacks.append(callback)

    def fire(self, event):
        for callback in self._callbacks:
            callback(event)


@contextmanager
def open_file_or_name(file: Union[TextIOWrapper, str], *args, **kwargs):
    file_wrapper = file
    if isinstance(file, str):
        file_wrapper = open(file, *args, **kwargs)

    yield file_wrapper

    if isinstance(file, str):
        file_wrapper.close()


def map_recursive(data, func: Callable):
    if torch.is_tensor(data):
        return func(data)
    if isinstance(data, (list, tuple, set)):
        return type(data)((map_recursive(datum, func) for datum in data))
    if isinstance(data, dict):
        return {key: map_recursive(value, func) for key, value in data.items()}
    return data


def to_device(data, device: torch.device):
    return map_recursive(data, lambda x: x.to(device))


def flatten(data):
    if isinstance(data, dict):
        data = list(data.values())
    if isinstance(data, (tuple, list)):
        return sum([flatten(elem) for elem in data], [])
    return [data]


def linear_from_params(weight: Tensor, bias: Optional[Tensor] = None) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        layer.weight.data = weight
        layer.bias.data = bias

    return layer


def reset_model(model: nn.Module):
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
    return model
