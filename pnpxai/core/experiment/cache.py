from typing import Any, Optional, Union
import torch
from pnpxai.utils import to_device


class ExperimentCache:
    __EXPLAINER_KEY = "explainer"
    __METRIC_KEY = "metric"
    __POSTPROCESSOR_KEY = "postprocessor"
    __EVALUATION_KEY = "evaluation"
    __DATA_KEY = "data"

    def __init__(self, cache_device: Optional[Union[torch.device, str]] = None):
        cpu_device = torch.device("cpu")
        if isinstance(cache_device, str):
            cache_device = torch.device(cache_device)
        self._device = cache_device if cache_device is not None else cpu_device

        self._global_cache = {}

    def to_device(self, x):
        return to_device(x, self._device)

    def get_output(self, data_id: int) -> Optional[Any]:
        key = self._get_key(data_id)
        return self._global_cache.get(key, None)

    def get_explanation(self, data_id: int, explainer_id: int) -> Optional[Any]:
        key = self._get_key(data_id, explainer_id)
        return self._global_cache.get(key, None)

    def get_evaluation(
        self,
        data_id: int,
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int
    ) -> Optional[Any]:
        key = self._get_key(data_id, explainer_id, postprocessor_id, metric_id)
        return self._global_cache.get(key, None)

    def set_output(self, data_id: int, output: Any):
        key = self._get_key(data_id)
        self._global_cache[key] = self.to_device(output)

    def set_explanation(self, data_id: int, explainer_id: int, explanation: Any):
        key = self._get_key(data_id, explainer_id)
        self._global_cache[key] = self.to_device(explanation)

    def set_evaluation(
        self,
        data_id: int,
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
        evaluation: Any
    ):
        key = self._get_key(data_id, explainer_id, postprocessor_id, metric_id)
        self._global_cache[key] = self.to_device(evaluation)

    def _get_key(
        self,
        data_id: int,
        explainer_id: Optional[int] = None,
        postprocessor_id: Optional[int] = None,
        metric_id: Optional[int] = None
    ):
        key = f"{self.__DATA_KEY}_{data_id}"
        if explainer_id is None:
            return key

        key = f"{key}.{self.__EXPLAINER_KEY}_{explainer_id}"
        if metric_id is None:
            return key

        return f"{key}.{self.__POSTPROCESSOR_KEY}_{postprocessor_id}.{self.__EVALUATION_KEY}_{metric_id}"
