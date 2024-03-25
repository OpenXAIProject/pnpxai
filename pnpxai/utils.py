import random
from io import TextIOWrapper
from contextlib import contextmanager
from typing import Sequence, Callable, Any, Union

import numpy as np
import torch


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


def to_device(data, device: torch.device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, (list, tuple, set)):
        return type(data)((to_device(datum, device) for datum in data))
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    return data
