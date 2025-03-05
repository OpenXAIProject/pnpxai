from abc import ABC
import copy
import torch


class UtilFunction(ABC):
    def __init__(self, *args, **kwargs):
        pass

    # def __repr__(self):
    #     return "{}({!r})".format(self.__class__.__name__, self.__dict__)

    def __repr__(self):
        kwargs_repr = ', '.join('{}={}'.format(key, value) for key, value in self.__dict__.items())
        return "{}({})".format(self.__class__.__name__, kwargs_repr)

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
