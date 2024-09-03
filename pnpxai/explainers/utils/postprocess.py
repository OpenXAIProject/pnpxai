from typing import Optional, Dict, Callable

import copy
from torch import Tensor
from pnpxai.explainers.utils.base import UtilFunction
from pnpxai.explainers.utils.function_selectors import FunctionSelector


def sumpos(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.sum(channel_dim).clamp(min=0)


def sumabs(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.sum(channel_dim).abs()


def l1norm(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.abs().sum(channel_dim)


def maxnorm(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.abs().max(channel_dim)[0]


def l2norm(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.pow(2).sum(channel_dim).sqrt()


def l2normsq(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.pow(2).sum(channel_dim)


def possum(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.clamp(min=0).sum(channel_dim)


def posmaxnorm(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.clamp(min=0).max(channel_dim)[0]


def posl2norm(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.clamp(min=0).pow(2).sum(channel_dim).sqrt()


def posl2normsq(attrs: Tensor, channel_dim: int) -> Tensor:
    return attrs.clamp(min=0).pow(2).sum(channel_dim)


def identity(attrs: Tensor, *args, **kwargs) -> Tensor:
    return attrs


class PoolingFunction(UtilFunction):
    """
    A base class for pooling functions that aggregate or summarize attribution data across 
    certain dimensions. Pooling functions are often used to reduce the dimensionality of 
    attributions and highlight important features.

    Parameters:
        channel_dim (int):
            The dimension of the input channels. This dimension is used by the pooling function 
            to perform aggregation operations correctly.

    Notes:
        - `PoolingFunction` is intended to be subclassed. Concrete pooling methods should 
          inherit from this class and implement the actual pooling logic.
        - The pooling operation should be compatible with the `channel_dim` provided during 
          initialization.
    """
    
    def __init__(self, channel_dim: int):
        super().__init__()
        self.channel_dim = channel_dim


class SumPos(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return sumpos(attrs, self.channel_dim)


class SumAbs(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return sumabs(attrs, self.channel_dim)


class L1Norm(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return l1norm(attrs, self.channel_dim)


class MaxNorm(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return maxnorm(attrs, self.channel_dim)


class L2Norm(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return l2norm(attrs, self.channel_dim)


class L2NormSquare(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return l2normsq(attrs, self.channel_dim)


class PosSum(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return possum(attrs, self.channel_dim)


class PosMaxNorm(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return posmaxnorm(attrs, self.channel_dim)


class PosL2Norm(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return posl2norm(attrs, self.channel_dim)


class PosL2NormSquare(PoolingFunction):
    def __init__(self, channel_dim):
        super().__init__(channel_dim)

    def __call__(self, attrs: Tensor):
        return posl2normsq(attrs, self.channel_dim)


class Identity(UtilFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs: Tensor):
        return identity(inputs)


POOLING_FUNCTIONS_FOR_IMAGE = {
    'sumpos': SumPos,
    'sumabs': SumAbs,
    'l1norm': L1Norm,
    'maxnorm': MaxNorm,
    'l2norm': L2Norm,
    'l2normsq': L2NormSquare,
    'possum': PosSum,
    'posmaxnorm': PosMaxNorm,
    'posl2norm': PosL2Norm,
    'posl2normsq': PosL2NormSquare,
}

POOLING_FUNCTIONS_FOR_TEXT = POOLING_FUNCTIONS_FOR_IMAGE

POOLING_FUNCTIONS_FOR_TIME_SERIES = {'identity': Identity}

POOLING_FUNCTIONS = {
    **POOLING_FUNCTIONS_FOR_IMAGE,
    **POOLING_FUNCTIONS_FOR_TEXT,
    **POOLING_FUNCTIONS_FOR_TIME_SERIES,
}


def minmax(attrs: Tensor):
    agg_dims = tuple(range(1, attrs.ndim))
    attrs_min = attrs.amin(agg_dims, keepdim=True)
    attrs_max = attrs.amax(agg_dims, keepdim=True)
    return (attrs - attrs_min) / (attrs_max - attrs_min)


class NormalizationFunction(UtilFunction):
    """
    A base class for normalization functions that adjust or scale attribution data. Normalization 
    functions are typically used to bring the data into a specific range or format for better 
    interpretation or visualization.

    Notes:
        - `NormalizationFunction` is designed to be subclassed. Concrete normalization methods 
          should inherit from this class and implement the actual normalization logic.
        - Subclasses can override the `__init__` method to accept additional parameters required 
          for their specific normalization operations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MinMax(NormalizationFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, attrs: Tensor):
        return minmax(attrs)


NORMALIZATION_FUNCTIONS_FOR_IMAGE = {
    'minmax': MinMax,
    'identity': Identity,
}

NORMALIZATION_FUNCTIONS_FOR_TEXT = NORMALIZATION_FUNCTIONS_FOR_IMAGE

NORMALIZATION_FUNCTIONS_FOR_TIME_SERIES = {
    'identity': Identity,
}

NORMALIZATION_FUNCTIONS = {
    **NORMALIZATION_FUNCTIONS_FOR_IMAGE,
    **NORMALIZATION_FUNCTIONS_FOR_TEXT,
    **NORMALIZATION_FUNCTIONS_FOR_TIME_SERIES,
}


class PostProcessor(UtilFunction):
    """
    A class that applies a series of post-processing steps to the output of an attribution method.
    This includes pooling and normalization functions to refine and transform the attributions.

    Parameters:
        pooling_fn (PoolingFunction):
            A function used to perform pooling on the attributions. Pooling typically involves 
            aggregating or summarizing the attributions over certain dimensions.
        normalization_fn (NormalizationFunction):
            A function used to normalize the pooled attributions. Normalization typically involves 
            scaling or adjusting the attributions to a certain range or format.

    Methods:
        from_name(pooling_method: str, normalization_method: str, channel_dim: int) -> PostProcessor:
            Creates a `PostProcessor` instance using the specified method names for pooling and 
            normalization, and the channel dimension. This is a convenience method for instantiating 
            `PostProcessor` with predefined methods.

        __call__(attrs):
            Applies the pooling and normalization functions to the given attributions. The input 
            attributions are first pooled and then normalized.

        get_tunables() -> Dict[str, Tuple[Type, Dict[str, Any]]]:
            Returns a dictionary of tunable parameters for the `PostProcessor`. The dictionary 
            includes the functions for pooling and normalization along with their default parameters.
    """
    
    def __init__(
        self,
        pooling_fn: PoolingFunction,
        normalization_fn: NormalizationFunction,
    ):
        self.pooling_fn = pooling_fn
        self.normalization_fn = normalization_fn

    @classmethod
    def from_name(
        cls,
        pooling_method: str,
        normalization_method: str,
        channel_dim: int,
    ):
        return cls(
            pooling_fn=POOLING_FUNCTIONS[pooling_method](channel_dim),
            normalization_fn=NORMALIZATION_FUNCTIONS[normalization_method](),
        )

    def __call__(self, attrs):
        pooled = self.pooling_fn(attrs)
        normalized = self.normalization_fn(pooled)
        return normalized

    def get_tunables(self):
        return {
            'pooling_fn': (PoolingFunction, {}),
            'normalization_fn': (NormalizationFunction, {}),
        }
