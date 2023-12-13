import warnings
from typing import List, Any, Dict, Type, Callable, Optional, Sequence, Union

from torch import Tensor
from torch.utils.data import DataLoader, Subset, Dataset

from pnpxai.explainers import Explainer, ExplainerWArgs, AVAILABLE_EXPLAINERS
from pnpxai.evaluator import XaiEvaluator
from pnpxai.core._types import DataSource, Model
from pnpxai.core.experiment.run import Run


def default_input_extractor(x):
    return x[0]


def default_target_extractor(x):
    return x[1]


class Experiment:
    def __init__(
        self,
        model: Model,
        data: DataSource,
        explainers: Sequence[Union[ExplainerWArgs, Explainer]],
        evaluator: XaiEvaluator = None,
        task: str = "image",
        input_extractor: Optional[Callable[[Any], Any]] = None,
        target_extractor: Optional[Callable[[Any], Any]] = None,
        input_visualizer: Optional[Callable[[Any], Any]] = None,
    ):
        self.model = model
        self.data = data
        self.evaluator = evaluator

        self.explainers_w_args: List[ExplainerWArgs] = self.preprocess_explainers(explainers) \
            if explainers is not None \
            else explainers

        self.input_extractor = input_extractor \
            if input_extractor is not None \
            else default_input_extractor
        self.target_extractor = target_extractor \
            if target_extractor is not None \
            else default_target_extractor
        self.input_visualizer = input_visualizer
        self.task = task
        self.runs: List[Run] = []

    def preprocess_explainers(self, explainers: Sequence[Union[ExplainerWArgs, Explainer]]) -> List[ExplainerWArgs]:
        return [
            explainer
            if isinstance(explainer, ExplainerWArgs)
            else ExplainerWArgs(explainer)
            for explainer in explainers
        ]

    @property
    def available_explainers(self) -> List[Type[Explainer]]:
        return list(map(lambda explainer: type(explainer.explainer), self.explainers_w_args))

    def add_explainer(
        self,
        explainer_type: Type[Explainer],
        attribute_kwargs: Optional[Dict[str, Any]] = None,
    ):
        attribute_kwargs = attribute_kwargs or {}
        explainer_w_args = ExplainerWArgs(
            explainer_type(self.model),
            kwargs=attribute_kwargs
        )
        self.explainers_w_args.append(explainer_w_args)

    def remove_explainer(self, idx: int):
        return self.explainers_w_args.pop(idx)

    def get_explainers_by_ids(self, explainer_ids: Optional[Sequence[int]] = None) -> List[ExplainerWArgs]:
        print([exp.explainer for exp in self.explainers_w_args])
        return [self.explainers_w_args[idx] for idx in explainer_ids] if explainer_ids is not None else self.explainers_w_args

    def get_data_by_ids(self, data_ids: Optional[Sequence[int]] = None) -> List[Any]:
        if data_ids is None:
            return self.data
        if isinstance(self.data, DataLoader):
            duplicated_params = [
                'num_workers', 'collate_fn', 'pin_memory', 'drop_last', 'timeout',
                'worker_init_fn', 'multiprocessing_context', 'generator', 'persistent_workers', 'pin_memory_device'
            ]
            batch_size = min(self.data.batch_size, len(data_ids)) \
                if self.data.batch_size is not None else None
            print("BS:", batch_size)
            data = self.data.__class__(
                dataset=Subset(self.data.dataset, data_ids), shuffle=False, batch_size=batch_size,
                **{param: getattr(self.data, param) for param in duplicated_params}
            )
        elif isinstance(self.data, Dataset):
            data = Subset(self.data, data_ids)
        else:
            data = self.data[data_ids]

        return data

    def run(
        self,
        data_ids: Optional[Sequence[int]] = None,
        explainer_ids: Optional[Sequence[int]] = None
    ) -> 'Experiment':
        explainers = self.get_explainers_by_ids(explainer_ids)
        data = self.get_data_by_ids(data_ids)
        runs = []

        for explainer in explainers:
            run = Run(
                data=data,
                input_extractor=self.input_extractor,
                target_extractor=self.target_extractor,
                explainer=explainer,
                evaluator=self.evaluator,
            )
            run.execute()
            runs.append(run)

        self.runs = runs
        return self

    def visualize(self):
        visualizations = [
            run.visualize(task=self.task)
            for run in self.runs
        ]
        return visualizations

    def rank_by_metrics(self):
        pass

    @property
    def is_image_task(self):
        return self.task == 'image'

    @property
    def is_batched(self):
        return isinstance(self.data, DataLoader) and self.data.batch_size > 1
