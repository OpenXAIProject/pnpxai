from typing import Optional, List, Callable, Type, Any, Dict
from abc import ABC, abstractmethod

from pnpxai.explainers.utils import UtilFunction
from pnpxai.explainers.utils.postprocess import (
    PostProcessor,
    POOLING_FUNCTIONS,
    POOLING_FUNCTIONS_FOR_IMAGE,
    POOLING_FUNCTIONS_FOR_TEXT,
    POOLING_FUNCTIONS_FOR_TIME_SERIES,
    NORMALIZATION_FUNCTIONS_FOR_IMAGE,
    NORMALIZATION_FUNCTIONS_FOR_TEXT,
    NORMALIZATION_FUNCTIONS_FOR_TIME_SERIES,
    PoolingFunction,
    NormalizationFunction
)
from pnpxai.explainers.utils.function_selectors import FunctionSelector
from pnpxai.explainers.utils.baselines import (
    BaselineFunction,
    BASELINE_FUNCTIONS,
    BASELINE_FUNCTIONS_FOR_IMAGE,
    BASELINE_FUNCTIONS_FOR_TEXT,
    BASELINE_FUNCTIONS_FOR_TIME_SERIES,
)
from pnpxai.explainers.utils.feature_masks import (
    FeatureMaskFunction,
    FEATURE_MASK_FUNCTIONS,
    FEATURE_MASK_FUNCTIONS_FOR_IMAGE,
    FEATURE_MASK_FUNCTIONS_FOR_TEXT,
    FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES,
)
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
    AVAILABLE_EXPLAINERS
)


class Modality(ABC):
    """
    An abstract class describing modality-specific workflow. The class is used to define both default and available
    explainers, baselines, feature masks, pooling methods, and normalization methods for the modality.

    Parameters:
        channel_dim (int): Target sequence dimension.
        baseline_fn_selector (Optional[FunctionSelector]): Selector of baselines for the modality's explainers. If None selected, all BASELINE_FUNCTIONS will be used.
        feature_mask_fn_selector (Optional[FunctionSelector]): Selector of feature masks for the modality's explainers. If None selected, all FEATURE_MASK_FUNCTIONS will be used.
        pooling_fn_selector (Optional[FunctionSelector]): Selector of pooling methods for the modality's explainers. If None selected, all POOLING_FUNCTIONS will be used.
        normalization_fn_selector (Optional[FunctionSelector]): Selector of normalization methods for the modality's explainers. If None selected, all NORMALIZATION_FUNCTIONS_FOR_IMAGE will be used.

    Attributes:
        EXPLAINERS (Tuple[Explainer]): Tuple of all available explainers.
    """

    # Copies the tuple without preserving the reference
    EXPLAINERS = tuple(iter(AVAILABLE_EXPLAINERS))

    def __init__(
        self,
        channel_dim: int,
        baseline_fn_selector: Optional[FunctionSelector] = None,
        feature_mask_fn_selector: Optional[FunctionSelector] = None,
        pooling_fn_selector: Optional[FunctionSelector] = None,
        normalization_fn_selector: Optional[FunctionSelector] = None,
        **kwargs
    ):
        self.channel_dim = channel_dim
        self.baseline_fn_selector = baseline_fn_selector or FunctionSelector(BASELINE_FUNCTIONS)
        self.feature_mask_fn_selector = feature_mask_fn_selector or FunctionSelector(FEATURE_MASK_FUNCTIONS)
        self.pooling_fn_selector = pooling_fn_selector or FunctionSelector(POOLING_FUNCTIONS)
        self.normalization_fn_selector = normalization_fn_selector or FunctionSelector(NORMALIZATION_FUNCTIONS_FOR_IMAGE)

    @abstractmethod    
    def get_default_feature_mask_fn(self) -> Callable:
        """
        Defines default baseline function for the modality's explainers.

        Returns:
            BaselineFunction: Zeros baseline function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_baseline_fn(self) -> Callable:
        """
        Defines default feature mask function for the modality's explainers.

        Returns:
            FeatureMaskFunction: No Mask baseline function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_default_postprocessors(self) -> List[Callable]:
        """
        Defines default post-processors list for the modality's explainers.

        Returns:
            List[PostProcessor]: Identity PostProcessors.
        """
        raise NotImplementedError
    
    def map_fn_selector(self, method_type: Type[Any]) -> Dict[Type[UtilFunction], callable]:
        """
        Selects custom optimizable hyperparameter functions.

        Returns:
            Dict[Type[UtilFunction], callable]: Identity PostProcessors.
        """
        return {
            BaselineFunction: self.baseline_fn_selector,
            FeatureMaskFunction: self.feature_mask_fn_selector,
            PoolingFunction: self.pooling_fn_selector,
            NormalizationFunction: self.normalization_fn_selector,
        }.get(method_type, None)


class ImageModality(Modality):
    """
    An extension of Modality class for Image domain with automatic explainers and evaluation metrics recommendation.

    Parameters:
        channel_dim (int): Target sequence dimension.
        baseline_fn_selector (Optional[FunctionSelector]): Selector of baselines for the modality's explainers. If None selected, BASELINE_FUNCTIONS_FOR_TIME_SERIES will be used.
        feature_mask_fn_selector (Optional[FunctionSelector]): Selector of feature masks for the modality's explainers. If None selected, FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES will be used.
        pooling_fn_selector (Optional[FunctionSelector]): Selector of pooling methods for the modality's explainers. If None selected, POOLING_FUNCTIONS_FOR_TIME_SERIES will be used.
        normalization_fn_selector (Optional[FunctionSelector]): Selector of normalization methods for the modality's explainers. If None selected, NORMALIZATION_FUNCTIONS_FOR_TIME_SERIES will be used.
    """
    def __init__(
        self,
        channel_dim: int = 1,
        baseline_fn_selector: Optional[FunctionSelector] = None,
        feature_mask_fn_selector: Optional[FunctionSelector] = None,
        pooling_fn_selector: Optional[FunctionSelector] = None,
        normalization_fn_selector: Optional[FunctionSelector] = None,
    ):
        super(ImageModality, self).__init__(
            channel_dim,
            baseline_fn_selector=baseline_fn_selector or FunctionSelector(
                data=BASELINE_FUNCTIONS_FOR_IMAGE,
                default_kwargs={'dim': channel_dim},
            ),
            feature_mask_fn_selector=feature_mask_fn_selector or FunctionSelector(
                data=FEATURE_MASK_FUNCTIONS_FOR_IMAGE
            ),
            pooling_fn_selector=pooling_fn_selector or FunctionSelector(
                data=POOLING_FUNCTIONS_FOR_IMAGE,
                default_kwargs={'channel_dim': channel_dim},
            ),
            normalization_fn_selector=normalization_fn_selector or FunctionSelector(
                data=NORMALIZATION_FUNCTIONS_FOR_IMAGE
            ),
        )

    def get_default_baseline_fn(self) -> BaselineFunction:
        """
        Defines default baseline function for the modality's explainers.

        Returns:
            BaselineFunction: Zeros baseline function.
        """
        return self.baseline_fn_selector.select('zeros')

    def get_default_feature_mask_fn(self) -> FeatureMaskFunction:
        """
        Defines default feature mask function for the modality's explainers.

        Returns:
            FeatureMaskFunction: Felzenszwalb baseline function.
        """
        return self.feature_mask_fn_selector.select('felzenszwalb', scale=250)

    def get_default_postprocessors(self) -> List[PostProcessor]:
        """
        Defines default post-processors list for the modality's explainers.

        Returns:
            List[PostProcessor]: All available PostProcessors.
        """
        return [
            PostProcessor(
                pooling_fn=self.pooling_fn_selector.select(pm),
                normalization_fn=self.normalization_fn_selector.select(nm),
            ) for pm in self.pooling_fn_selector.choices
            for nm in self.normalization_fn_selector.choices
        ]


class TextModality(Modality):
    """
    An extension of Modality class for Text domain with automatic explainers and evaluation metrics recommendation.

    Parameters:
        channel_dim (int): Target sequence dimension.
        baseline_fn_selector (Optional[FunctionSelector]): Selector of baselines for the modality's explainers. If None selected, BASELINE_FUNCTIONS_FOR_TEXT will be used.
        feature_mask_fn_selector (Optional[FunctionSelector]): Selector of feature masks for the modality's explainers. If None selected, FEATURE_MASK_FUNCTIONS_FOR_TEXT will be used.
        pooling_fn_selector (Optional[FunctionSelector]): Selector of pooling methods for the modality's explainers. If None selected, POOLING_FUNCTIONS_FOR_TEXT will be used.
        normalization_fn_selector (Optional[FunctionSelector]): Selector of normalization methods for the modality's explainers. If None selected, NORMALIZATION_FUNCTIONS_FOR_TEXT will be used.
    """
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

    def __init__(
        self,
        channel_dim: int = -1,
        mask_token_id: int = 0,
        baseline_fn_selector: Optional[FunctionSelector] = None,
        feature_mask_fn_selector: Optional[FunctionSelector] = None,
        pooling_fn_selector: Optional[FunctionSelector] = None,
        normalization_fn_selector: Optional[FunctionSelector] = None,
    ):
        super(TextModality, self).__init__(
            channel_dim,
            baseline_fn_selector=baseline_fn_selector or FunctionSelector(
                data=BASELINE_FUNCTIONS_FOR_TEXT,
                default_kwargs={'token_id': mask_token_id},
            ),
            feature_mask_fn_selector=feature_mask_fn_selector or FunctionSelector(
                data=FEATURE_MASK_FUNCTIONS_FOR_TEXT
            ),
            pooling_fn_selector=pooling_fn_selector or FunctionSelector(
                data=POOLING_FUNCTIONS_FOR_TEXT,
                default_kwargs={'channel_dim': channel_dim},
            ),
            normalization_fn_selector=normalization_fn_selector or FunctionSelector(
                data=NORMALIZATION_FUNCTIONS_FOR_TEXT,
            ),
        )
        self.mask_token_id = mask_token_id

    def get_default_baseline_fn(self) -> BaselineFunction:
        """
        Defines default baseline function for the modality's explainers.

        Returns:
            BaselineFunction: Token baseline function.
        """
        return self.baseline_fn_selector.select('token')

    def get_default_feature_mask_fn(self) -> FeatureMaskFunction:
        """
        Defines default feature mask function for the modality's explainers.

        Returns:
            FeatureMaskFunction: No Mask baseline function.
        """
        return self.feature_mask_fn_selector.select('no_mask_1d')

    def get_default_postprocessors(self) -> List[PostProcessor]:
        """
        Defines default post-processors list for the modality's explainers.

        Returns:
            List[PostProcessor]: All PostProcessors.
        """
        return [
            PostProcessor(
                pooling_fn=self.pooling_fn_selector.select(pm),
                normalization_fn=self.normalization_fn_selector.select(nm),
            ) for pm in self.pooling_fn_selector.choices
            for nm in self.normalization_fn_selector.choices
        ]


class TimeSeriesModality(Modality):
    """
    An extension of Modality class for Time Series domain with automatic explainers and evaluation metrics recommendation.

    Parameters:
        channel_dim (int): Target sequence dimension.
        baseline_fn_selector (Optional[FunctionSelector]): Selector of baselines for the modality's explainers. If None selected, BASELINE_FUNCTIONS_FOR_TIME_SERIES will be used.
        feature_mask_fn_selector (Optional[FunctionSelector]): Selector of feature masks for the modality's explainers. If None selected, FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES will be used.
        pooling_fn_selector (Optional[FunctionSelector]): Selector of pooling methods for the modality's explainers. If None selected, POOLING_FUNCTIONS_FOR_TIME_SERIES will be used.
        normalization_fn_selector (Optional[FunctionSelector]): Selector of normalization methods for the modality's explainers. If None selected, NORMALIZATION_FUNCTIONS_FOR_TIME_SERIES will be used.
    """
    def __init__(
        self,
        channel_dim: int = -1,
        baseline_fn_selector: Optional[FunctionSelector] = None,
        feature_mask_fn_selector: Optional[FunctionSelector] = None,
        pooling_fn_selector: Optional[FunctionSelector] = None,
        normalization_fn_selector: Optional[FunctionSelector] = None,
    ):
        super(TimeSeriesModality, self).__init__(
            channel_dim,
            baseline_fn_selector=baseline_fn_selector or FunctionSelector(
                data=BASELINE_FUNCTIONS_FOR_TIME_SERIES, # [zeros, mean]
                default_kwargs={'dim': channel_dim},
            ),
            feature_mask_fn_selector=feature_mask_fn_selector or FunctionSelector(
                data=FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES,
            ),
            pooling_fn_selector=pooling_fn_selector or FunctionSelector(
                data=POOLING_FUNCTIONS_FOR_TIME_SERIES, # [identity]
                default_kwargs={'channel_dim': channel_dim},
            ),
            normalization_fn_selector=normalization_fn_selector or FunctionSelector(
                data=NORMALIZATION_FUNCTIONS_FOR_TIME_SERIES, # [identity]
            ),
        )

    def get_default_baseline_fn(self) -> BaselineFunction:
        """
        Defines default baseline function for the modality's explainers.

        Returns:
            BaselineFunction: Zeros baseline function.
        """
        return self.baseline_fn_selector.select('zeros')

    def get_default_feature_mask_fn(self) -> FeatureMaskFunction:
        """
        Defines default feature mask function for the modality's explainers.

        Returns:
            FeatureMaskFunction: No Mask baseline function.
        """
        return self.feature_mask_fn_selector.select('no_mask_2d')

    def get_default_postprocessors(self) -> List[PostProcessor]:
        """
        Defines default post-processors list for the modality's explainers.

        Returns:
            List[PostProcessor]: Identity PostProcessors.
        """
        return [
            PostProcessor(
                pooling_fn=self.pooling_fn_selector.select(pm),
                normalization_fn=self.normalization_fn_selector.select(nm),
            ) for pm in self.pooling_fn_selector.choices
            for nm in self.normalization_fn_selector.choices
        ]
