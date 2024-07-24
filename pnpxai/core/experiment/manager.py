from typing import Any, List, Optional, Sequence, Union, Type, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, Dataset

from pnpxai.core.experiment.cache import ExperimentCache
from pnpxai.core._types import DataSource
from pnpxai.explainers.base import Explainer
from pnpxai.metrics.base import Metric
# from pnpxai.explainers._explainer import Explainer, Explainer
# from pnpxai.evaluator import Metric


class ExperimentManager:
    def __init__(
        self,
        data: DataSource,
        explainers: Sequence[Explainer],
        metrics: Sequence[Metric],
        cache_device: Optional[Union[torch.device, str]] = None,
    ):
        self._data = data
        self._metrics = metrics
        self._explainers = explainers
        # self._explainers_w_args: List[Explainer] = self.preprocess_explainers(explainers) \
        #     if explainers is not None \
        #     else explainers

        self._cache = ExperimentCache(cache_device)
        self.set_config()

    # def preprocess_explainers(self, explainers: Sequence[Explainer]) -> List[Explainer]:
    #     return [
    #         explainer
    #         if isinstance(explainer, Explainer)
    #         else Explainer(explainer)
    #         for explainer in explainers
    #     ]

    def set_config(
        self,
        data_ids: Optional[Sequence[int]] = None,
        explainer_ids: Optional[Sequence[int]] = None,
        metric_ids: Optional[Sequence[int]] = None,
    ):
        self.set_data_ids(data_ids)
        self.set_explainer_ids(explainer_ids)
        self.set_metric_ids(metric_ids)

    def get_data_ids(self) -> Sequence[int]:
        return self._data_ids

    def set_data_ids(self, data_ids: Optional[Sequence[int]] = None):
        self._data_ids = data_ids if data_ids is not None else list(
            range(self._all_data_len))

    def set_explainer_ids(self, explainer_ids: Optional[Sequence[int]] = None):
        self._explainer_ids = explainer_ids if explainer_ids is not None else list(
            range(len(self._explainers))
            # range(len(self._explainers_w_args))
        )

    def set_metric_ids(self, metric_ids: Optional[Sequence[int]] = None):
        self._metric_ids = metric_ids if metric_ids is not None else list(
            range(len(self._metrics))
        )

    def get_explainers(self) -> Tuple[List[Explainer], List[int]]:
        return self._get_explainers_by_ids(self._explainer_ids), self._explainer_ids

    def _get_explainers_by_ids(self, explainer_ids: Optional[Sequence[int]] = None) -> List[Explainer]:
        # return [self._explainers_w_args[idx] for idx in explainer_ids] if explainer_ids is not None else self._explainers_w_args
        return [self._explainers[idx] for idx in explainer_ids] if explainer_ids is not None else self._explainers

    def get_metrics(self) -> Tuple[List[Metric], List[int]]:
        return self._get_metrics_by_ids(self._metric_ids), self._metric_ids

    def _get_metrics_by_ids(self, metric_ids: Optional[Sequence[int]] = None) -> List[Metric]:
        return [self._metrics[idx] for idx in metric_ids] if metric_ids is not None else self._metrics

    def get_data_to_process_for_explainer(self, explainer_id: int) -> Tuple[DataSource, List[int]]:
        data_ids = [
            idx for idx in self._data_ids if self._cache.get_explanation(idx, explainer_id) is None
        ]
        return self._get_data_by_ids(data_ids), data_ids

    def get_data_to_process_for_metric(self, explainer_id: int, metric_id: int) -> Tuple[DataSource, List[int]]:
        data_ids = [
            idx for idx in self._data_ids if self._cache.get_evaluation(idx, explainer_id, metric_id) is None
        ]
        return self._get_data_by_ids(data_ids), data_ids

    def get_data_to_predict(self) -> Tuple[DataSource, List[int]]:
        data_ids = [
            idx for idx in self._data_ids if self._cache.get_output(idx) is None
        ]
        return self._get_data_by_ids(data_ids), data_ids

    def get_data(self, data_ids: Optional[Sequence[int]] = None) -> Tuple[DataSource, List[int]]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        return self._get_data_by_ids(data_ids), data_ids

    def get_all_data(self) -> DataSource:
        return self._get_data_by_ids()

    def get_valid_explanations(self, explainer_id: int, data_ids: Optional[Sequence[int]] = None) -> Tuple[DataSource, List[int]]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        explanations = self.get_flat_explanations(explainer_id, data_ids)
        valid_explanations = []
        valid_data_ids = []

        for idx, explanation in zip(data_ids, explanations):
            if explanation is None:
                continue
            valid_explanations.append(explanation)
            valid_data_ids.append(idx)

        if self.is_batched:
            return self._copy_data_loader(valid_explanations, self._data.__class__, do_collate=False), valid_data_ids

        return valid_explanations, valid_data_ids

    def get_flat_explanations(self, explainer_id: int, data_ids: Optional[Sequence[int]] = None) -> Sequence[Tensor]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        return [self._cache.get_explanation(idx, explainer_id) for idx in data_ids]

    def get_flat_evaluations(self, explainer_id: int, metric_id: int, data_ids: Optional[Sequence[int]] = None) -> Sequence[Tensor]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        return [self._cache.get_evaluation(idx, explainer_id, metric_id) for idx in data_ids]

    def get_flat_outputs(self, data_ids: Optional[Sequence[int]] = None) -> Tuple[DataSource, List[int]]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        return [self._cache.get_output(idx) for idx in data_ids]

    def save_explanations(self, explanations: DataSource, data: DataSource, data_ids: Sequence[int], explainer_id: int):
        explanations = self.flatten_if_batched(explanations, data)
        assert len(explanations) == len(data_ids)

        for idx, explanation in zip(data_ids, explanations):
            self._cache.set_explanation(idx, explainer_id, explanation)

    def save_evaluations(self, evaluations: DataSource, data: DataSource, data_ids: Sequence[int], explainer_id: int, metric_id: int):
        evaluations = self.flatten_if_batched(evaluations, data)
        assert len(evaluations) == len(data_ids)

        for idx, evaluation in zip(data_ids, evaluations):
            self._cache.set_evaluation(
                idx, explainer_id, metric_id, evaluation)

    def save_outputs(self, outputs: DataSource, data: DataSource, data_ids: Sequence[int]):
        outputs = self.flatten_if_batched(outputs, data)
        assert len(outputs) == len(data_ids)

        for idx, output in zip(data_ids, outputs):
            self._cache.set_output(idx, output)

    def _get_batch_size(self, data: DataSource) -> Optional[int]:
        if torch.is_tensor(data):
            return len(data)
        for item in data:
            item_len = self._get_batch_size(item)
            if item_len is not None:
                return item_len
        return None

    def flatten_if_batched(self, values: DataSource, length_reference: DataSource):
        if not self.is_batched:
            return values
        assert len(values) == len(length_reference)

        flattened = []
        for batch, reference in zip(values, length_reference):
            if batch is None:
                reference_len = self._get_batch_size(reference) or 1
                batch = [None] * reference_len
            if isinstance(batch, Tuple):
                batch = zip(*[list(b) for b in batch])
            flattened.extend(list(batch))
        return flattened

    def _get_data_by_ids(self, data_ids: Optional[Sequence[int]] = None) -> List[Any]:
        if data_ids is None:
            return self._data
        if isinstance(self._data, DataLoader):
            data = self._copy_data_loader(
                data=Subset(self._data.dataset, data_ids),
                data_loader=self._data.__class__
            )
        elif isinstance(self._data, Dataset):
            data = Subset(self._data, data_ids)
        else:
            try:
                data = self._data[data_ids]
            except:
                data = [self._data[idx] for idx in data_ids]

        return data

    def _copy_data_loader(self, data: DataSource, data_loader: Type[DataLoader], do_collate=True):
        duplicated_params = [
            'batch_size', 'num_workers', 'pin_memory', 'drop_last', 'timeout',
            'worker_init_fn', 'multiprocessing_context', 'generator', 'persistent_workers', 'pin_memory_device'
        ]
        collate_fn = getattr(self._data, 'collate_fn') if do_collate else None
        return data_loader(
            dataset=data, shuffle=False, collate_fn=collate_fn,
            **{param: getattr(self._data, param) for param in duplicated_params}
        )

    @property
    def all_explainers(self) -> Sequence[Explainer]:
        # return self._explainers_w_args
        return self._explainers

    @property
    def all_metrics(self) -> Sequence[Metric]:
        return self._metrics

    @property
    def has_metrics(self):
        return len(self._metrics) > 0

    @property
    def has_explanations(self):
        for explainer_id in self._explainer_ids:
            _, valid_data_ids = self.get_valid_explanations(explainer_id)
            if len(valid_data_ids) > 0:
                return True
        return False

    @property
    def is_batched(self):
        return isinstance(self._data, DataLoader) and self._data.batch_size is not None

    @property
    def _all_data_len(self):
        return len(self._data.dataset if isinstance(self._data, DataLoader) else self._data)

    @property
    def _all_data_ids(self):
        return range(self._all_data_len)

    @property
    def _available_data_ids(self):
        return self._all_data_ids if self._data_ids is None else self._data_ids
