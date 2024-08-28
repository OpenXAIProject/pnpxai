import copy
from torch import Tensor


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


RELEVANCE_POOLING_METHODS = {
    'sumpos': sumpos,
    'sumabs': sumabs,
    'l1norm': l1norm,
    'maxnorm': maxnorm,
    'l2norm': l2norm,
    'l2normsq': l2normsq,
    'possum': possum,
    'posmaxnorm': posmaxnorm,
    'posl2norm': posl2norm,
    'posl2normsq': posl2normsq,
    'identity': identity,
}

ALL_RELEVANCE_POOLING_METHODS = {
    **RELEVANCE_POOLING_METHODS,
    'identity': identity,
}


def relevance_pooling(attrs: Tensor, channel_dim: int, method='l2normsq'):
    return ALL_RELEVANCE_POOLING_METHODS[method](attrs, channel_dim)


def minmax_normalization(attrs: Tensor):
    agg_dims = tuple(range(1, attrs.ndim))
    attrs_min = attrs.amin(agg_dims, keepdim=True)
    attrs_max = attrs.amax(agg_dims, keepdim=True)

    return (attrs - attrs_min) / (attrs_max - attrs_min)


RELEVANCE_NORMALIZATION_METHODS = {
    'minmax': minmax_normalization,
    'identity': identity,
}

ALL_RELEVANCE_NORMALIZATION_METHODS = {
    **RELEVANCE_NORMALIZATION_METHODS,
    'identity': identity,
}


def normalize_relevance(pooled_attr: Tensor, method='minmax'):
    return RELEVANCE_NORMALIZATION_METHODS[method](pooled_attr)


def postprocess_attr(
    attr: Tensor,
    channel_dim: int,
    pooling_method: str = 'l2normsq',
    normalization_method: str = 'minmax',
):
    pooled_attr = relevance_pooling(attr, channel_dim, method=pooling_method)
    normalized_attr = normalize_relevance(
        pooled_attr, method=normalization_method)

    return normalized_attr


class PostProcessor:
    def __init__(
        self,
        pooling_method='l2normsq',
        normalization_method='minmax',
        channel_dim=-1,
    ):
        self.pooling_method = pooling_method
        self.normalization_method = normalization_method
        self.channel_dim = channel_dim

    def __repr__(self):
        return f"PostProcessor(pooling_method={self.pooling_method}, normalization_method={self.normalization_method})"

    def set_channel_dim(self, channel_dim: int):
        self.channel_dim = channel_dim
        return self

    def set_kwargs(self, **kwargs):
        clone = self.copy()
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(clone, k, v)
        return clone

    def copy(self):
        return copy.copy(self)

    def pool_attributions(self, attrs):
        return relevance_pooling(attrs, channel_dim=self.channel_dim, method=self.pooling_method)

    def normalize_attributions(self, attrs):
        return normalize_relevance(attrs, method=self.normalization_method)

    def __call__(self, attrs):
        return postprocess_attr(
            attrs,
            channel_dim=self.channel_dim,
            pooling_method=self.pooling_method,
            normalization_method=self.normalization_method
        )

    def copy(self):
        return PostProcessor(
            pooling_method=self.pooling_method,
            normalization_method=self.normalization_method,
            channel_dim=self.channel_dim,
        )
