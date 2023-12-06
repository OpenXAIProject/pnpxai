import random
import numpy as np
import torch
from typing import Sequence, Callable, Any


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_to_string(object):
    return object.__class__.__name__


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
