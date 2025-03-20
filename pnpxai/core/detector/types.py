from importlib import util
from typing import Union

from torch import nn
from zennit.types import SubclassMeta
from zennit.types import Convolution as ZennitConvolution
from zennit.types import BatchNorm as ZennitBatchNorm
from zennit.types import Activation as ZennitActivation
from zennit.types import AvgPool as ZennitAvgPool
from zennit.types import MaxPool as ZennitMaxPool


class Linear(metaclass=SubclassMeta):
    '''Abstract base class that describes linear modules.'''
    __subclass__ = (
        nn.Linear,
    )


class Convolution(metaclass=SubclassMeta):
    '''Abstract base class that describes convolution modules'''
    __subclass__ = (
        ZennitConvolution,
    )


class Pool(metaclass=SubclassMeta):
    __subclass__ = (
        ZennitAvgPool,
        ZennitMaxPool,
    )


class BatchNorm(metaclass=SubclassMeta):
    '''Abstract base class that describes batchnorm modules'''
    __subclass__ = (
        ZennitBatchNorm,
    )


class Activation(metaclass=SubclassMeta):
    '''Abstract base class that describes activation modules'''
    __subclass__ = (
        ZennitActivation,
    )


class LSTM(metaclass=SubclassMeta):
    '''Abstract base class that describes lstm modules.'''
    __subclass__ = (
        nn.LSTM,
    )


class RNN(metaclass=SubclassMeta):
    '''Abstract base class that describes linear modules.'''
    __subclass__ = (
        nn.RNN,
    )


def _get_attention_subclasses():
    subclasses = [nn.MultiheadAttention]
    if util.find_spec("timm"):
        import timm
        subclasses.append(timm.models.vision_transformer.Attention)
    if util.find_spec("transformers"):
        from transformers.models.bert.modeling_bert import BertAttention as BertAttentionOfTransformers
        from transformers.models.visual_bert.modeling_visual_bert import VisualBertAttention as VisualBertAttentionOfTransformers
        from transformers.models.vilt.modeling_vilt import ViltAttention as ViltAttentionOfTransformers
        subclasses.append(BertAttentionOfTransformers)
        subclasses.append(VisualBertAttentionOfTransformers)
        subclasses.append(ViltAttentionOfTransformers)
    return tuple(subclasses)


class Attention(metaclass=SubclassMeta):
    '''Abstract base class that describes attention modules'''
    __subclass__ = _get_attention_subclasses()


class Embedding(metaclass=SubclassMeta):
    __subclass__ = (
        nn.Embedding,
        nn.EmbeddingBag,
    )


ModuleType = Union[Linear, Convolution, RNN, LSTM, Attention, Pool, Embedding]
