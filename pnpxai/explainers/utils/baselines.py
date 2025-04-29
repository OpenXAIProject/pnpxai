from typing import Literal, Union, Literal

import torch
import torchvision.transforms.functional as TF
from pnpxai.explainers.utils.base import UtilFunction

BaselineMethod = Literal['zeros', 'invert', 'gaussian_blur', 'token']


class BaselineFunction(UtilFunction):
    """
    A base class for baseline functions used in attribution methods. Baseline functions are 
    used to define a reference or baseline value against which attributions are compared.
    This is typically used to understand the effect of different inputs on the model's predictions.

    Notes:
        - `BaselineFunction` is intended to be subclassed. Concrete baseline functions should 
          inherit from this class and implement the actual baseline logic.
        - Subclasses can override the `__init__` method to accept additional parameters required 
          for their specific baseline operations.
    """    
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_method(cls, method, **kwargs):
        baseline_fn = BASELINE_FUNCTIONS.get(method, None)
        if baseline_fn is None:
            raise ValueError
        return baseline_fn(**kwargs)


class TokenBaselineFunction(BaselineFunction):
    def __init__(self, token_id, **kwargs):
        super().__init__()
        self.token_id = token_id

    def __call__(self, inputs: torch.Tensor):
        return torch.ones_like(inputs, dtype=torch.long) * self.token_id


class ZeroBaselineFunction(BaselineFunction):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs: torch.Tensor):
        return torch.zeros_like(inputs)


class MeanBaselineFunction(BaselineFunction):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim

    def __call__(self, inputs: torch.Tensor):
        size = inputs.size(self.dim)
        target_dim = self.dim % inputs.ndim
        # print(self.dim, inputs.ndim, target_dim)
        repeat = tuple(size if d == target_dim else 1 for d in range(inputs.dim()))
        outputs = torch.mean(inputs, dim=self.dim, keepdim=True).repeat(*repeat)
        return outputs


class InvertBaselineFunction(BaselineFunction):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, inputs: torch.Tensor):
        return TF.invert(inputs)


class GaussianBlurBaselineFunction(BaselineFunction):
    def __init__(
        self,
        kernel_size_x: int = 3,
        kernel_size_y: int = 3,
        sigma_x: float = .5,
        sigma_y: float = .5,
        **kwargs
    ):
        super().__init__()
        self.kernel_size_x = kernel_size_x
        self.kernel_size_y = kernel_size_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def __call__(self, inputs: torch.Tensor):
        return TF.gaussian_blur(
            inputs,
            kernel_size=[self.kernel_size_x, self.kernel_size_y],
            sigma=[self.sigma_x, self.sigma_y],
        )

    def get_tunables(self):
        return {
            'kernel_size_x': (int, {'low': 1, 'high': 11, 'step': 2}),
            'kernel_size_y': (int, {'low': 1, 'high': 11, 'step': 2}),
            'sigma_x': (float, {'low': .05, 'high': 2., 'step': .05}),
            'sigma_y': (float, {'low': .05, 'high': 2., 'step': .05}),
        }


BaselineMethodOrFunction = Union[BaselineMethod, BaselineFunction]

BASELINE_FUNCTIONS_FOR_IMAGE = {
    'zeros': ZeroBaselineFunction,
    'mean': MeanBaselineFunction,
    'invert': InvertBaselineFunction,
    'gaussian_blur': GaussianBlurBaselineFunction,
}

BASELINE_FUNCTIONS_FOR_TEXT = {
    'token': TokenBaselineFunction,
}

BASELINE_FUNCTIONS_FOR_TIME_SERIES = {
    'zeros': ZeroBaselineFunction,
    'mean': MeanBaselineFunction,
}

BASELINE_FUNCTIONS = {
    **BASELINE_FUNCTIONS_FOR_IMAGE,
    **BASELINE_FUNCTIONS_FOR_TEXT,
    **BASELINE_FUNCTIONS_FOR_TIME_SERIES,
}