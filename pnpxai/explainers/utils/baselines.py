from typing import Literal, Union

import torch
import torchvision.transforms.functional as TF
from pnpxai.explainers.base import Tunable
from pnpxai.explainers.types import TunableParameter
from pnpxai.explainers.utils.base import UtilFunction


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


class TokenBaselineFunction(BaselineFunction):
    def __init__(self, token_id):
        super().__init__()
        self.token_id = token_id

    def __call__(self, inputs: torch.Tensor):
        return torch.ones_like(inputs, dtype=torch.long) * self.token_id


class ZeroBaselineFunction(BaselineFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: torch.Tensor):
        return torch.zeros_like(inputs)


class MeanBaselineFunction(BaselineFunction):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, inputs: torch.Tensor):
        size = inputs.size(self.dim)
        repeat = tuple(size if d == self.dim else 1 for d in range(inputs.dim()))
        return torch.mean(inputs, dim=self.dim, keepdim=True).repeat(*repeat)


class InvertBaselineFunction(BaselineFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: torch.Tensor):
        return TF.invert(inputs)


class GaussianBlurBaselineFunction(BaselineFunction, Tunable):
    def __init__(
        self,
        kernel_size_x: int = 3,
        kernel_size_y: int = 3,
        sigma_x: float = .5,
        sigma_y: float = .5,
    ):
        BaselineFunction.__init__(self)
        self.kernel_size_x = TunableParameter(
            name='kernel_size_x',
            current_value=kernel_size_x,
            dtype=int,
            is_leaf=True,
            space={'low': 1, 'high': 11, 'step': 2},
        )
        self.kernel_size_y = TunableParameter(
            name='kernel_size_y',
            current_value=kernel_size_y,
            dtype=int,
            is_leaf=True,
            space={'low': 1, 'high': 11, 'step': 2},
        )
        self.sigma_x = TunableParameter(
            name='sigma_x',
            current_value=sigma_x,
            dtype=float,
            is_leaf=True,
            space={'low': .05, 'high': 2., 'step': .05},
        )
        self.sigma_y = TunableParameter(
            name='sigma_y',
            current_value=sigma_y,
            dtype=float,
            is_leaf=True,
            space={'low': .05, 'high': 2., 'step': .05},
        )
        Tunable.__init__(self)
        self.register_tunable_params([
            self.kernel_size_x, self.kernel_size_y,
            self.sigma_x, self.sigma_y,
        ])

    def __call__(self, inputs: torch.Tensor):
        return TF.gaussian_blur(
            inputs,
            kernel_size=[
                self.kernel_size_x.current_value,
                self.kernel_size_y.current_value,
            ],
            sigma=[self.sigma_x.current_value, self.sigma_y.current_value],
        )


BASELINE_FUNCTIONS = {  # (dtype, ndims): {available util functions}
    (float, 2): {
        'zeros': ZeroBaselineFunction,
        'mean': MeanBaselineFunction,
    },
    (float, 4): {
        'zeros': ZeroBaselineFunction,
        'mean': MeanBaselineFunction,
        'invert': InvertBaselineFunction,
        'gaussian_blur': GaussianBlurBaselineFunction,
    },
    (int, 2): {
        'token': TokenBaselineFunction,
    },
}
