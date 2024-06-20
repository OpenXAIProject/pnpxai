from torch import Tensor

RELEVANCE_POOLING_METHODS = {
    'sumpos': lambda attr, channel_dim: attr.sum(channel_dim).clamp(min=0),
    'sumabs': lambda attr, channel_dim: attr.sum(channel_dim).abs(),
    'l1norm': lambda attr, channel_dim: attr.abs().sum(channel_dim),
    'maxnorm': lambda attr, channel_dim: attr.abs().max(channel_dim)[0],
    'l2norm': lambda attr, channel_dim: attr.pow(2).sum(channel_dim).sqrt(),
    'l2normsq': lambda attr, channel_dim: attr.pow(2).sum(channel_dim),
    'possum': lambda attr, channel_dim: attr.clamp(min=0).sum(channel_dim),
    'posmaxnorm': lambda attr, channel_dim: attr.clamp(min=0).max(channel_dim)[0],
    'posl2norm': lambda attr, channel_dim: attr.clamp(min=0).pow(2).sum(channel_dim).sqrt(),
    'posl2normsq': lambda attr, channel_dim: attr.clamp(min=0).pow(2).sum(channel_dim),
}

def relevance_pooling(attr: Tensor, channel_dim: int, method='sumpos'):
    return RELEVANCE_POOLING_METHODS[method](attr, channel_dim)