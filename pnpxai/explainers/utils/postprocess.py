from torch import Tensor
from pnpxai.explainers.utils.base import UtilFunction
from pnpxai.explainers.base import Tunable
from pnpxai.explainers.types import TunableParameter


def sumpos(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.sum(pooling_dim).clamp(min=0)


def sumabs(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.sum(pooling_dim).abs()


def l1norm(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.abs().sum(pooling_dim)


def maxnorm(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.abs().max(pooling_dim)[0]


def l2norm(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.pow(2).sum(pooling_dim).sqrt()


def l2normsq(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.pow(2).sum(pooling_dim)


def possum(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.clamp(min=0).sum(pooling_dim)


def posmaxnorm(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.clamp(min=0).max(pooling_dim)[0]


def posl2norm(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.clamp(min=0).pow(2).sum(pooling_dim).sqrt()


def posl2normsq(attrs: Tensor, pooling_dim: int) -> Tensor:
    return attrs.clamp(min=0).pow(2).sum(pooling_dim)


def identity(attrs: Tensor, *args, **kwargs) -> Tensor:
    return attrs


class PoolingFunction(UtilFunction):
    """
    A base class for pooling functions that aggregate or summarize attribution data across 
    certain dimensions. Pooling functions are often used to reduce the dimensionality of 
    attributions and highlight important features.

    Parameters:
        pooling_dim (int):
            The dimension of the input channels. This dimension is used by the pooling function 
            to perform aggregation operations correctly.

    Notes:
        - `PoolingFunction` is intended to be subclassed. Concrete pooling methods should 
          inherit from this class and implement the actual pooling logic.
        - The pooling operation should be compatible with the `pooling_dim` provided during 
          initialization.
    """
    
    def __init__(self, pooling_dim: int):
        super().__init__()
        self.pooling_dim = pooling_dim


class SumPos(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return sumpos(attrs, self.pooling_dim)


class SumAbs(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return sumabs(attrs, self.pooling_dim)


class L1Norm(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return l1norm(attrs, self.pooling_dim)


class MaxNorm(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return maxnorm(attrs, self.pooling_dim)


class L2Norm(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return l2norm(attrs, self.pooling_dim)


class L2NormSquare(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return l2normsq(attrs, self.pooling_dim)


class PosSum(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return possum(attrs, self.pooling_dim)


class PosMaxNorm(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return posmaxnorm(attrs, self.pooling_dim)


class PosL2Norm(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return posl2norm(attrs, self.pooling_dim)


class PosL2NormSquare(PoolingFunction):
    def __init__(self, pooling_dim):
        super().__init__(pooling_dim)

    def __call__(self, attrs: Tensor):
        return posl2normsq(attrs, self.pooling_dim)


class Identity(UtilFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, inputs: Tensor):
        return identity(inputs)


POOLING_FUNCTIONS = {
    (float, 2): {
        'identity': Identity,
    },
    (float, 4): {
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
    },
    (int, 2): {
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
    },
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


NORMALIZATION_FUNCTIONS = {
    (float, 2): {
        'identity': Identity,
    },
    (float, 4): {
        'identity': Identity,
        'minmax': MinMax,
    },
    (int, 2): {
        'identity': Identity,
        'minmax': MinMax,
    },
}


class PostProcessor(UtilFunction, Tunable):
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
        from_name(pooling_method: str, normalization_method: str, pooling_dim: int) -> PostProcessor:
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
        modality,
        pooling_method=None,
        normalization_method=None,
    ):
        UtilFunction.__init__(self)
        self.modality = modality
        self._pooling_method = TunableParameter(
            name='pooling_method',
            current_value=pooling_method or modality.util_functions['pooling_fn'].choices[0],
            dtype=str,
            is_leaf=True,
            space={'choices': modality.util_functions['pooling_fn'].choices}
        )
        self._normalization_method = TunableParameter(
            name='normalization_method',
            current_value=normalization_method or modality.util_functions['normalization_fn'].choices[0],
            dtype=str,
            is_leaf=True,
            space={'choices': modality.util_functions['normalization_fn'].choices}
        )
        Tunable.__init__(self)
        self.register_tunable_params([
            self._pooling_method, self._normalization_method])

    @property
    def pooling_method(self):
        return self._pooling_method.current_value

    @property
    def normalization_method(self):
        return self._normalization_method.current_value

    @property
    def pooling_fn(self):
        return self.modality.util_functions['pooling_fn'].select(self.pooling_method)

    @property
    def normalization_fn(self):
        return self.modality.util_functions['normalization_fn'].select(self.normalization_method)

    def __call__(self, attrs):
        pooled = self.pooling_fn(attrs)
        normalized = self.normalization_fn(pooled)
        return normalized
