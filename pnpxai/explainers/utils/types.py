from typing import Tuple, Union
from pnpxai.explainers.utils.baselines import BaselineFunction
from pnpxai.explainers.utils.feature_masks import FeatureMaskFunction
from pnpxai.explainers.utils.postprocess import PostProcessor


BaselineFunctionOrTupleOfBaselineFunctions = Union[
    BaselineFunction,
    Tuple[BaselineFunction, ...]
]

FeatureMaskFunctionOrTupleOfFeatureMaskFunctions = Union[
    FeatureMaskFunction,
    Tuple[FeatureMaskFunction, ...]
]

PostProcessorOrTupleOfPostProcessors = Union[
    PostProcessor,
    Tuple[PostProcessor, ...]
]
