import copy
from torch import Tensor
from pnpxai.explainers.utils.postprocess.methods import relevance_pooling, normalize_relevance


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
