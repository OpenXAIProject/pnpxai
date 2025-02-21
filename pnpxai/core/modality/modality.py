from typing import Optional, Any
from collections import defaultdict

from pnpxai.explainers.utils.postprocess import (
    POOLING_FUNCTIONS,
    NORMALIZATION_FUNCTIONS,
)
from pnpxai.explainers.utils.function_selectors import FunctionSelector
from pnpxai.explainers.utils.baselines import (
    BASELINE_FUNCTIONS,
    TokenBaselineFunction,
    MeanBaselineFunction,
)
from pnpxai.explainers.utils.feature_masks import FEATURE_MASK_FUNCTIONS


class Modality:
    UTIL_FUNCTIONS = {
        'baseline_fn': BASELINE_FUNCTIONS,
        'feature_mask_fn': FEATURE_MASK_FUNCTIONS,
        'pooling_fn': POOLING_FUNCTIONS,
        'normalization_fn': NORMALIZATION_FUNCTIONS,
    }

    def __init__(
        self,
        dtype: Any,
        ndims: Any,
        pooling_dim: Optional[int] = None,
        mask_token_id: Optional[int] = None,
    ):
        self.dtype = dtype
        self.ndims = ndims
        self.pooling_dim = pooling_dim
        self.mask_token_id = mask_token_id

        self._util_functions = defaultdict(FunctionSelector)
        self._set_util_functions()
    
    @property
    def dtype_key(self):
        # TODO: more elegant way
        if 'float' in str(self.dtype):
            return float
        if 'int' in str(self.dtype) or 'long' in str(self.dtype):
            return int
        raise ValueError(f'No matched key for {self.dtype}')

    @property
    def util_functions(self):
        return self._util_functions

    def _set_util_functions(self):
        for space_key, spaces in self.UTIL_FUNCTIONS.items():
            space = spaces[(self.dtype_key, self.ndims)]
            for fn_key, fn_type in space.items():
                # set non-varying kwargs such as pooling dim or token id
                if (
                    space_key == 'pooling_fn'
                    and 'pooling_dim' not in self._util_functions[space_key].default_kwargs
                ):
                    self._util_functions[space_key].add_default_kwargs(
                        'pooling_dim', self.pooling_dim)
                if fn_type is TokenBaselineFunction:
                    self._util_functions[space_key].add_default_kwargs(
                        'token_id', self.mask_token_id, choice='token')
                if fn_type is MeanBaselineFunction:
                    dim = self.pooling_dim or 0
                    self._util_functions[space_key].add_default_kwargs(
                        'dim', dim, choice='mean')

                # add fn_type to space
                self._util_functions[space_key].add(fn_key, fn_type)

