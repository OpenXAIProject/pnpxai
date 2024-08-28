from typing import Union, Tuple, Optional, List
from abc import ABC
from optuna import Trial
from pnpxai.evaluator.optimizer.utils import generate_param_key
from pnpxai.explainers.utils.feature_masks import (
    FeatureMaskFunction,
    FEATURE_MASK_FUNCTIONS,
    FEATURE_MASK_FUNCTIONS_FOR_IMAGE,
    FEATURE_MASK_FUNCTIONS_FOR_TEXT,
    FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES,
    suggest_tunable_feature_masks
)
from pnpxai.explainers.utils.baselines import (
    BaselineFunction,
    BASELINE_METHODS_FOR_IMAGE,
    BASELINE_METHODS_FOR_TEXT,
    BASELINE_METHODS_FOR_TIME_SERIES,
    BASELINE_METHODS,
    suggest_tunable_baselines
)
from pnpxai.explainers.utils.postprocess.postprocess import PostProcessor
from pnpxai.explainers.utils.postprocess.methods import RELEVANCE_POOLING_METHODS, RELEVANCE_NORMALIZATION_METHODS
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
    FEATURE_MASKS = tuple(FEATURE_MASK_FUNCTIONS.keys())
    BASELINES = tuple(BASELINE_METHODS.keys())

    def __init__(self, channel_dim: Union[int, Tuple[int]]):
        self.channel_dim = channel_dim

    def get_default_feature_mask_fn(self):
        return FeatureMaskFunction(method='no_mask')

    def get_default_baseline_fn(self, *args, **kwargs):
        return BaselineFunction(method='zeros')

    def get_default_postprocessors(self) -> List[PostProcessor]:
        return [
            PostProcessor(
                pooling_method=pm,
                normalization_method=nm,
                channel_dim=self.channel_dim
            ) for pm in self.PP_RELEVANCE_POOLING_METHODS
            for nm in self.PP_RELEVANCE_NORMALIZATION_METHODS
        ]

    def suggest_tunable_post_processors(self, trial: Trial, key: Optional[str] = None):
        return {
            'pooling_method': trial.suggest_categorical(
                generate_param_key(key, 'pooling_method'),
                choices=self.PP_RELEVANCE_POOLING_METHODS,
            ),
            'normalization_method': trial.suggest_categorical(
                generate_param_key(key, 'normalization_method'),
                choices=self.PP_RELEVANCE_NORMALIZATION_METHODS,
            ),
        }

    def suggest_tunable_feature_masks(self, trial: Trial, key: Optional[str] = None):
        return suggest_tunable_feature_masks(self.FEATURE_MASKS, trial, key)

    def suggest_tunable_baselines(self, trial: Trial, key: Optional[str] = None):
        return suggest_tunable_baselines(self.BASELINES, trial, key)


class ImageModality(Modality):
    FEATURE_MASKS = tuple(FEATURE_MASK_FUNCTIONS_FOR_IMAGE.keys())
    BASELINES = tuple(BASELINE_METHODS_FOR_IMAGE.keys())

    def __init__(self, channel_dim: int = 1):
        super(ImageModality, self).__init__(channel_dim)

    def get_default_feature_mask_fn(self):
        return FeatureMaskFunction(method='felzenszwalb', scale=250)


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
        AttentionRollout,
        TransformerAttribution
    )
    PP_RELEVANCE_POOLING_METHODS = tuple(
        k for k in RELEVANCE_POOLING_METHODS.keys()
        if k != 'identity'
    )
    FEATURE_MASKS = tuple(FEATURE_MASK_FUNCTIONS_FOR_TEXT.keys())
    BASELINES = tuple(BASELINE_METHODS_FOR_TEXT.keys())

    def __init__(self, channel_dim: int = -1):
        super(TextModality, self).__init__(channel_dim)

    def get_default_baseline_fn(self, mask_token_id: Optional[int] = None):
        return BaselineFunction(method='mask_token', token_id=mask_token_id)


class ImageTextModality(ImageModality, TextModality):
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
    )

    def __init__(self, channel_dim: Tuple[int] = (1, -1)):
        super().__init__(channel_dim)

    def get_default_feature_mask_fn(self):
        return (
            ImageModality.get_default_feature_mask_fn(self),
            TextModality.get_default_feature_mask_fn(self),
        )

    def get_default_baseline_fn(self, mask_token_id: Optional[int] = None):
        return (
            ImageModality.get_default_baseline_fn(self),
            TextModality.get_default_baseline_fn(self, mask_token_id),
        )

    def get_default_postprocessors(self):
        return [
            tuple(PostProcessor(
                pooling_method=pm,
                normalization_method=nm,
                channel_dim=d,
            ) for d in self.channel_dim
            ) for pm in RELEVANCE_POOLING_METHODS
            for nm in RELEVANCE_NORMALIZATION_METHODS
        ]


class TimeSeriesModality(Modality):
    PP_RELEVANCE_POOLING_METHODS = ('identity',)
    PP_RELEVANCE_NORMALIZATION_METHODS = ('identity',)
    BASELINES = tuple(BASELINE_METHODS_FOR_TIME_SERIES.keys())
    FEATURE_MASKS = tuple(FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES.keys())

    def __init__(self, channel_dim: int = -1):
        super(TimeSeriesModality, self).__init__(channel_dim)

    def get_default_feature_mask_fn(self):
        return FeatureMaskFunction(method='no_mask')
