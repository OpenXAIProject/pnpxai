from typing import Union, Tuple, Optional, List, Sequence
from abc import ABC, abstractmethod
import re

from optuna import Trial
from pnpxai.evaluator.optimizer.utils import generate_param_key
from pnpxai.explainers.utils.postprocess import (
    PostProcessor,
    RELEVANCE_POOLING_METHODS,
    RELEVANCE_NORMALIZATION_METHODS,
)
from pnpxai.explainers.utils.function_selectors import BaselineFunctionSelector, FeatureMaskFunctionSelector
from pnpxai.explainers import (
    Gradient,
    GradientXInput,
    SmoothGrad,
    VarGrad,
    IntegratedGradients,
    LRPUniformEpsilon,
    LRPEpsilonPlus,
    LRPEpsilonGammaBox,
    LRPEpsilonAlpha2Beta1,
    KernelShap,
    Lime,
    AttentionRollout,
    TransformerAttribution,
    AVAILABLE_EXPLAINERS
)


class Modality(ABC):
    # Copies the tuple without preserving the reference
    EXPLAINERS = tuple(iter(AVAILABLE_EXPLAINERS))
    PP_RELEVANCE_POOLING_METHODS = tuple(RELEVANCE_POOLING_METHODS.keys())
    PP_RELEVANCE_NORMALIZATION_METHODS = tuple(
        RELEVANCE_NORMALIZATION_METHODS.keys()
    )

    def __init__(self, channel_dim: Union[int, Tuple[int]]):
        self.channel_dim = channel_dim

    @property
    def name(self):
        nm = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()
        return nm.split('_')[0]

    @abstractmethod    
    def get_default_feature_mask_fn(self):
        raise NotImplementedError

    @abstractmethod
    def get_default_baseline_fn(self, *args, **kwargs):
        raise NotImplementedError

    def get_default_postprocessors(self) -> List[PostProcessor]:
        return [
            PostProcessor(
                pooling_method=pm,
                normalization_method=nm,
                channel_dim=self.channel_dim
            ) for pm in self.PP_RELEVANCE_POOLING_METHODS
            for nm in self.PP_RELEVANCE_NORMALIZATION_METHODS
        ]

class ImageModality(Modality):
    def __init__(self, channel_dim: int = 1):
        super(ImageModality, self).__init__(channel_dim)

    def get_default_baseline_fn(self):
        return BaselineFunctionSelector(self).select('zeros')

    def get_default_feature_mask_fn(self):
        return FeatureMaskFunctionSelector(self).select('felzenszwalb', scale=250)


class TextModality(Modality):
    EXPLAINERS = (
        Gradient,
        GradientXInput,
        SmoothGrad,
        VarGrad,
        IntegratedGradients,
        LRPUniformEpsilon,
        LRPEpsilonPlus,
        LRPEpsilonGammaBox,
        LRPEpsilonAlpha2Beta1,
        KernelShap,
        Lime,
        # AttentionRollout,
        # TransformerAttribution
    )
    PP_RELEVANCE_POOLING_METHODS = tuple(
        k for k in RELEVANCE_POOLING_METHODS.keys()
        if k != 'identity'
    )

    def __init__(self, channel_dim: int = -1, mask_token_id: int = 0):
        super(TextModality, self).__init__(channel_dim)
        self.mask_token_id = mask_token_id

    def get_default_baseline_fn(self, mask_token_id: Optional[int] = None):
        return BaselineFunctionSelector(self).select('token', token_id=self.mask_token_id)

    def get_default_feature_mask_fn(self):
        return FeatureMaskFunctionSelector(self).select('no_mask_1d')


class TimeSeriesModality(Modality):
    PP_RELEVANCE_POOLING_METHODS = ('identity',)
    PP_RELEVANCE_NORMALIZATION_METHODS = ('identity',)

    def __init__(self, channel_dim: int = -1):
        super(TimeSeriesModality, self).__init__(channel_dim)

    def get_default_baseline_fn(self):
        return BaselineFunctionSelector(self).select('zeros')

    def get_default_feature_mask_fn(self):
        return FeatureMaskFunctionSelector(self).select('no_mask_1d')
