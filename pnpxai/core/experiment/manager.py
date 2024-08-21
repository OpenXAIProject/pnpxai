from typing import Any, List, Optional, Sequence, Union, Type, Tuple, Callable

import itertools

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, Dataset

from pnpxai.core.experiment.cache import ExperimentCache
from pnpxai.core._types import DataSource
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.postprocess import PostProcessor
from pnpxai.evaluator.metrics.base import Metric
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


def _index_combinations(*indices):
    return itertools.product(*indices)


class ExperimentManager:
    def __init__(
        self,
        data: DataSource,
        cache_device: Optional[Union[torch.device, str]] = None,
    ):
        self._data = data
        self.set_data_ids()
        self._explainers: List[Explainer] = []
        self._explainer_ids: List[int] = []
        self._postprocessors: List[Callable] =[]
        self._postprocessor_ids: List[int] = []
        self._metrics: List[Metric] = []
        self._metric_ids: List[int] = []

        self._cache = ExperimentCache(cache_device)

    def clear(self):
        self._explainers = []
        self._explainer_ids = []
        self._metrics = []
        self._metric_ids = []
        self._cache._global_cache = {}

    @property
    def data(self):
        return self._data

    @property
    def explainers(self):
        return self._explainers

    @property
    def postprocessors(self):
        return self._postprocessors

    @property
    def metrics(self):
        return self._metrics

    def set_data_ids(self, data_ids: Optional[Sequence[int]] = None):
        self._data_ids = data_ids if data_ids is not None else list(
            range(self._all_data_len))

    def cache_outputs(self, data_ids, outputs):
        assert len(data_ids) == len(outputs)
        for idx, output in zip(data_ids, outputs):
            self._cache.set_output(idx, output)

    def cache_explanations(
        self,
        explainer_id: int,
        data_ids: List[int],
        explanations,
    ):
        explanations = format_into_tuple(explanations)
        assert all(len(expl) == len(data_ids) for expl in explanations)
        for idx, *explanation in zip(data_ids, *explanations):
            explanation = format_into_tuple(explanation)
            self._cache.set_explanation(idx, explainer_id, explanation)

    def cache_evaluations(
        self,
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
        data_ids: List[int],
        evaluations: DataSource,
    ):
        evaluations = format_into_tuple(evaluations)
        assert all(len(ev) == len(data_ids) for ev in evaluations)
        for idx, *evaluation in zip(data_ids, *evaluations):
            evaluation = format_into_tuple(evaluation)
            self._cache.set_evaluation(
                idx, explainer_id, postprocessor_id, metric_id, evaluation)

    def get_data_ids(self) -> Sequence[int]:
        return self._data_ids

    def _get_data_by_ids(
        self,
        data_ids: Optional[Sequence[int]]=None,
    ) -> List[Any]:
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

    def get_data(
        self,
        data_ids: Optional[Sequence[int]]=None,
    ) -> Tuple[DataSource, List[int]]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        return self._get_data_by_ids(data_ids), data_ids

    ###
    def get_data_by_id(self, data_id: int):
        data = self._data.dataset[data_id]
        data = self._data.collate_fn([data])
        return tuple(d.squeeze(0) for d in data)

    def get_output_by_id(self, data_id: int):
        return self._cache.get_output(data_id)

    def get_explainer_by_id(self, explainer_id: int):
        return self._explainers[explainer_id]

    def get_explanation_by_id(self, data_id: int, explainer_id: int):
        cached = self._cache.get_explanation(data_id, explainer_id)
        if isinstance(cached, Tuple):
            return format_out_tuple_if_single(cached)
        return cached # This may be None

    def get_postprocessor_by_id(self, postprocessor_id: int):
        return self._postprocessors[postprocessor_id]

    def get_metric_by_id(self, metric_id: int):
        return self._metrics[metric_id]

    def get_evaluation_by_id(
        self,
        data_id: int,
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int
    ):
        cached = self._cache.get_evaluation(
            data_id, explainer_id, postprocessor_id, metric_id)
        if isinstance(cached, Tuple):
            return format_out_tuple_if_single(cached)
        return cached # This may be None

    def batch_data_by_ids(
        self,
        data_ids: List[int],
    ):
        batch = [self._data.dataset[idx] for idx in data_ids]
        batch = self._data.collate_fn(batch)
        return batch

    def batch_outputs_by_ids(self, data_ids: List[int]):
        batch = []
        for data_id in data_ids:
            output = self.get_output_by_id(data_id)
            if output is None:
                raise KeyError(f"Output for {data_id} does not exist in cache.")
            batch.append(output)
        return self._format_batch(batch)

    def batch_explainers_by_ids(self, explainer_ids: List[int]):
        return [self._explainers[idx] for idx in explainer_ids]

    def batch_explanations_by_ids(
        self,
        data_ids: List[int],
        explainer_id: int,
    ):
        batch = [
            self.get_explanation_by_id(data_id, explainer_id)
            for data_id in data_ids
        ]
        return self._format_batch(batch)

    def batch_evaluations_by_ids(
        self,
        data_ids: List[int],
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
    ):
        batch = [
            self.get_evaluation_by_id(
                data_id, explainer_id, postprocessor_id, metric_id)
            for data_id in data_ids
        ]
        return self._format_batch(batch)

    def _format_batch(self, batch):
        if isinstance(batch[0], Tuple):
            cnt = len(batch[0])
            batch = [[data[i] for data in batch] for i in range(cnt)]
        else:
            batch = [batch]
        formatted = *(self._stack_batch(b) for b in batch),
        formatted = format_out_tuple_if_single(formatted)
        return formatted

    def _stack_batch(self, single_batch):
        if torch.is_tensor(single_batch[0]):
            return torch.stack(single_batch)
        return single_batch            

    def add_explainer(self, explainer: Explainer) -> int:
        explainer_id = len(self._explainers)
        self._explainers.append(explainer)
        self._explainer_ids.append(explainer_id)
        return explainer_id

    def _get_explainers_by_ids(self, explainer_ids: Optional[Sequence[int]] = None) -> List[Explainer]:
        # return [self._explainers_w_args[idx] for idx in explainer_ids] if explainer_ids is not None else self._explainers_w_args
        return [self._explainers[idx] for idx in explainer_ids] if explainer_ids is not None else self._explainers

    def get_explainers(self, explainer_ids: Optional[List[int]]=None) -> Tuple[List[Explainer], List[int]]:
        explainer_ids = explainer_ids or self._explainer_ids
        return self._get_explainers_by_ids(explainer_ids), explainer_ids

    def add_postprocessor(self, postprocessor: Callable) -> int:
        postprocessor_id = len(self._postprocessors)
        self._postprocessors.append(postprocessor)
        self._postprocessor_ids.append(postprocessor_id)
        return postprocessor_id

    def _get_postprocessors_by_ids(self, postprocessor_ids: Optional[Sequence[int]]=None) -> List[Callable]:
        return [self._postprocessors[idx] for idx in postprocessor_ids] if postprocessor_ids is not None else self._postprocessors

    def get_postprocessors(self, postprocessor_ids: Optional[List[int]]=None) -> Tuple[List[Callable], List[int]]:
        postprocessor_ids = postprocessor_ids or self._postprocessor_ids
        return self._get_postprocessors_by_ids(postprocessor_ids), postprocessor_ids

    # metric
    def add_metric(self, metric: Metric) -> int:
        metric_id = len(self._metrics)
        self._metrics.append(metric)
        self._metric_ids.append(metric_id)
        return metric_id

    def _get_metrics_by_ids(self, metric_ids: Optional[Sequence[int]] = None) -> List[Metric]:
        return [self._metrics[idx] for idx in metric_ids] if metric_ids is not None else self._metrics

    def get_metrics(self, metric_ids: Optional[List[int]]=None) -> Tuple[List[Metric], List[int]]:
        metric_ids = metric_ids or self._metric_ids
        return self._get_metrics_by_ids(metric_ids), metric_ids

    # explanation
    def get_data_to_process_for_explainer(
        self,
        explainer_id: int,
        data_ids: Optional[List[int]] = None,
    ) -> Tuple[DataSource, List[int]]:
        data_ids = data_ids or self._data_ids
        data_ids = [
            idx for idx in data_ids if self._cache.get_explanation(idx, explainer_id) is None
        ]
        return self._get_data_by_ids(data_ids), data_ids

    def save_explanations(self, explanations: DataSource, data: DataSource, data_ids: Sequence[int], explainer_id: int):
        explanations = self.flatten_if_batched(explanations, data)
        assert len(explanations) == len(data_ids)
        for idx, explanation in zip(data_ids, explanations):
            self._cache.set_explanation(idx, explainer_id, explanation)

    def get_data_to_process_for_metric(
        self,
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
        data_ids: Optional[List[int]]=None,
    ) -> Tuple[DataSource, List[int]]:
        data_ids = data_ids or self._data_ids
        data_ids = [
            idx for idx in data_ids
            if self._cache.get_evaluation(idx, explainer_id, postprocessor_id, metric_id) is None
        ]
        return self._get_data_by_ids(data_ids), data_ids

    def get_data_to_predict(self, data_ids: List[int]) -> Tuple[DataSource, List[int]]:
        data_ids = [
            idx for idx in data_ids if self._cache.get_output(idx) is None
        ]
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

    def get_flat_evaluations(
        self,
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int,
        data_ids: Optional[Sequence[int]] = None
    ) -> Sequence[Tensor]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        return [self._cache.get_evaluation(
            idx, explainer_id, postprocessor_id, metric_id) for idx in data_ids]

    def get_flat_outputs(self, data_ids: Optional[Sequence[int]] = None) -> Tuple[DataSource, List[int]]:
        data_ids = data_ids if data_ids is not None else self._data_ids
        return [self._cache.get_output(idx) for idx in data_ids]

    def save_evaluations(
        self,
        evaluations: DataSource,
        data: DataSource,
        data_ids: Sequence[int],
        explainer_id: int,
        postprocessor_id: int,
        metric_id: int
    ):
        evaluations = self.flatten_if_batched(evaluations, data)
        assert len(evaluations) == len(data_ids)

        for idx, evaluation in zip(data_ids, evaluations):
            self._cache.set_evaluation(
                idx, explainer_id, metric_id, postprocessor_id, evaluation)

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
