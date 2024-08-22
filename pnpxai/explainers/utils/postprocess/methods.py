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


def relevance_pooling(attrs: Tensor, channel_dim: int, method='l2normsq'):
    return RELEVANCE_POOLING_METHODS[method](attrs, channel_dim)


def minmax_normalization(attrs: Tensor):
    agg_dims = tuple(range(1, attrs.ndim))
    attrs_min = attrs.amin(agg_dims, keepdim=True)
    attrs_max = attrs.amax(agg_dims, keepdim=True)

    return (attrs - attrs_min) / (attrs_max - attrs_min)


RELEVANCE_NORMALIZATION_METHODS = {
    'minmax': minmax_normalization,
    'identity': identity,
}


def normalize_relevance(pooled_attr: Tensor, method='minmax'):
    return RELEVANCE_NORMALIZATION_METHODS[method](pooled_attr)
