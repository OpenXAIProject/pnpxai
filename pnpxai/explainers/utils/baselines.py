from typing import Literal, Optional, Union, Literal, Sequence
import copy

import torch
import torchvision.transforms.functional as TF
from optuna.trial import Trial
from pnpxai.explainers.utils.base import UtilFunction
from pnpxai.evaluator.optimizer.utils import generate_param_key


BaselineMethod = Literal['zeros', 'invert', 'gaussian_blur', 'token']


class BaselineFunction(UtilFunction):
    def __init__(self):
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


class GaussianBlurBaselineFunction(BaselineFunction):
    def __init__(
        self,
        kernel_size_x: int = 3,
        kernel_size_y: int = 3,
        sigma_x: float = .5,
        sigma_y: float = .5,
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