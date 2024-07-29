import math
from torch import Tensor

RELEVANCE_POOLING_METHODS = {
    'sumpos': lambda attrs, channel_dim: attrs.sum(channel_dim).clamp(min=0),
    'sumabs': lambda attrs, channel_dim: attrs.sum(channel_dim).abs(),
    'l1norm': lambda attrs, channel_dim: attrs.abs().sum(channel_dim),
    'maxnorm': lambda attrs, channel_dim: attrs.abs().max(channel_dim)[0],
    'l2norm': lambda attrs, channel_dim: attrs.pow(2).sum(channel_dim).sqrt(),
    'l2normsq': lambda attrs, channel_dim: attrs.pow(2).sum(channel_dim),
    'possum': lambda attrs, channel_dim: attrs.clamp(min=0).sum(channel_dim),
    'posmaxnorm': lambda attrs, channel_dim: attrs.clamp(min=0).max(channel_dim)[0],
    'posl2norm': lambda attrs, channel_dim: attrs.clamp(min=0).pow(2).sum(channel_dim).sqrt(),
    'posl2normsq': lambda attrs, channel_dim: attrs.clamp(min=0).pow(2).sum(channel_dim),
}


def relevance_pooling(attrs: Tensor, channel_dim: int, method='l2normsq'):
    return RELEVANCE_POOLING_METHODS[method](attrs, channel_dim)


def minmax_normalization(attrs: Tensor):
    bsz, *size = attrs.size()
    attrs_min = attrs.view(-1, math.prod(size)).min(axis=1).values
    attrs_min = attrs_min[(...,)+(None,)*len(size)]
    attrs_max = attrs.view(-1, math.prod(size)).max(axis=1).values
    attrs_max = attrs_max[(...,)+(None,)*len(size)]
    return (attrs - attrs_min) / (attrs_max - attrs_min)


RELEVANCE_NORMALIZATION_METHODS = {
    'minmax': minmax_normalization,
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
    normalized_attr = normalize_relevance(pooled_attr, method=normalization_method)
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
        return f"PostProcessor(pooling_method='{self.pooling_method}', normalization_method='{self.normalization_method}', channel_dim={self.channel_dim})"

    def set_channel_dim(self, channel_dim: int):
        self.channel_dim = channel_dim
        return self

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


def all_postprocessors(channel_dim):
    if isinstance(channel_dim, int):
        return [PostProcessor(
            pooling_method=pm,
            normalization_method=nm,
            channel_dim=channel_dim
        ) for pm in RELEVANCE_POOLING_METHODS
        for nm in RELEVANCE_NORMALIZATION_METHODS]
    return [tuple(
        PostProcessor(
            pooling_method=pm,
            normalization_method=nm,
            channel_dim=channel_dim[d]
        ) for d in channel_dim
    ) for pm in RELEVANCE_POOLING_METHODS for nm in RELEVANCE_NORMALIZATION_METHODS]