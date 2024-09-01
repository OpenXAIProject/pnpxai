from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Dict

from pnpxai.explainers.utils.baselines import (
    BASELINE_FUNCTIONS,
    BASELINE_FUNCTIONS_FOR_IMAGE,
    BASELINE_FUNCTIONS_FOR_TEXT,
    BASELINE_FUNCTIONS_FOR_TIME_SERIES,
)
from pnpxai.explainers.utils.feature_masks import (
    FEATURE_MASK_FUNCTIONS,
    FEATURE_MASK_FUNCTIONS_FOR_IMAGE,
    FEATURE_MASK_FUNCTIONS_FOR_TEXT,
    FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES,
)


class FunctionSelector:
    def __init__(self, data: Optional[Dict[str, Callable]]=None):
        super().__init__()
        self._data = {} or data

    @property
    def choices(self):
        return list(self._data.keys())

    def add(self, key: str, value: Callable):
        self._data[key] = value
        return value

    def get(self, key: str):
        return self._data[key]

    def delete(self, key: str):
        return self._data.pop(key, None)

    def all(self):
        return [self.get(key) for key in self.choices]

    def select(self, key: str, **kwargs):
        fn_type = self.get(key)
        return fn_type(**kwargs)

    # @property
    # @abstractmethod
    # def choices(self) -> List[str]:
    #     raise NotImplementedError

    # @abstractmethod
    # def select(self, choice: str, **kwargs) -> Callable:
    #     raise NotImplementedError

    def get_tunables(self):
        return {'method': (list, {'choices': self.choices})}


# class BaselineFunctionSelector(FunctionSelector):
#     def __init__(self, modality: 'Modality'):
#         super().__init__(modality)

#     @property
#     def choices(self):
#         return {
#             'image': BASELINE_FUNCTIONS_FOR_IMAGE.keys(),
#             'text': BASELINE_FUNCTIONS_FOR_TEXT.keys(),
#             'time_series': BASELINE_FUNCTIONS_FOR_TIME_SERIES.keys(),
#         }.get(self.modality.name, None)

#     def select(self, choice, **kwargs):
#         data = {
#             'image': BASELINE_FUNCTIONS_FOR_IMAGE,
#             'text': BASELINE_FUNCTIONS_FOR_TEXT,
#             'time_series': BASELINE_FUNCTIONS_FOR_TIME_SERIES,
#         }.get(self.modality.name, None)
#         fn_type = data.get(choice)
#         return fn_type(**kwargs)


# class FeatureMaskFunctionSelector(FunctionSelector):
#     def __init__(self, modality: 'Modality'):
#         super().__init__(modality)

#     @property
#     def choices(self):
#         return {
#             'image': FEATURE_MASK_FUNCTIONS_FOR_IMAGE.keys(),
#             'text': FEATURE_MASK_FUNCTIONS_FOR_TEXT.keys(),
#             'time_series': FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES.keys(),
#         }.get(self.modality.name, None)

#     def select(self, choice, **kwargs):
#         data = {
#             'image': FEATURE_MASK_FUNCTIONS_FOR_IMAGE,
#             'text': FEATURE_MASK_FUNCTIONS_FOR_TEXT,
#             'time_series': FEATURE_MASK_FUNCTIONS_FOR_TIME_SERIES,
#         }.get(self.modality.name, None)
#         fn_type = data.get(choice)
#         return fn_type(**kwargs)


# class PostProcessorSelector(FunctionSelector):
#     def __init__(self, modality: 'Modality'):
#         super().__init__(modality)

#     @property
#     def choices(self):
#         return {
#             'image': postprocessors_for_
#         }