from abc import ABC
import copy
import torch


class UtilFunction(ABC):
    def __init__(self, **kwargs):
        pass

    def copy(self):
        return copy.copy(self)

    def set_kwargs(self, **kwargs):
        clone = self.copy()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(clone, k, v)
        return clone

    def __call__(self, inputs: torch.Tensor):
        return NotImplementedError

    def get_tunables(self):
        return {}

