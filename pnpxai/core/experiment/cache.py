from typing import Any, Optional
from typing import Any


class ExperimentCache:
    __EXPLAINER_KEY = "explainer"
    __EVALUATION_KEY = "evaluation"
    __DATA_KEY = "data"

    def __init__(self):
        self._global_cache = {}

    def get_output(self, data_id: int) -> Optional[Any]:
        key = self._get_key(data_id)
        return self._global_cache.get(key, None)

    def get_explanation(self, data_id: int, explainer_id: int) -> Optional[Any]:
        key = self._get_key(data_id, explainer_id)
        return self._global_cache.get(key, None)

    def get_evaluation(self, data_id: int, explainer_id: int, metric_id: int) -> Optional[Any]:
        key = self._get_key(data_id, explainer_id, metric_id)
        return self._global_cache.get(key, None)

    def set_output(self, data_id: int, output: Any):
        key = self._get_key(data_id)
        self._global_cache[key] = output

    def set_explanation(self, data_id: int, explainer_id: int, explanation: Any):
        key = self._get_key(data_id, explainer_id)
        self._global_cache[key] = explanation

    def set_evaluation(self, data_id: int, explainer_id: int, metric_id: int, evaluation: Any):
        key = self._get_key(data_id, explainer_id, metric_id)
        self._global_cache[key] = evaluation

    def _get_key(self, data_id: int, explainer_id: Optional[int] = None, metric_id: Optional[int] = None):
        key = f"{self.__DATA_KEY}_{data_id}"
        if explainer_id is None:
            return key

        key = f"{key}.{self.__EXPLAINER_KEY}_{explainer_id}"
        if metric_id is None:
            return key

        return f"{key}.{self.__EVALUATION_KEY}_{metric_id}"
