import warnings
from typing import List, Any, Dict, Type, Callable, Optional, Sequence, Union

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
        name: str,
        model: Model,
        data: DataSource,
        explainers: Optional[Sequence[Union[ExplainerWArgs, Explainer]]] = None,
        evaluator: XaiEvaluator = None,
        task: str = "classification",
        input_extractor: Optional[Callable[[Any], Any]] = None,
        target_extractor: Optional[Callable[[Any], Any]] = None,
    ):
        self.name = name
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
        self.task = task
        self.runs: List[Run] = []

    def preprocess_explainers(self, explainers: Optional[Sequence[Union[ExplainerWArgs, Explainer]]] = None) -> List[ExplainerWArgs]:
        if explainers is None:
            return AVAILABLE_EXPLAINERS

        return [
            explainer
            if isinstance(explainer, ExplainerWArgs)
            else ExplainerWArgs(explainer)
            for explainer in explainers
        ]

    @property
    def available_explainers(self) -> List[Type[Explainer]]:
        return list(map(lambda explainer: type(explainer.explainer), self.explainers_w_args))

    def __repr__(self):
        return f"<Experiment: {self.name}>"

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
        return self.explainers_w_args if explainer_ids is None else [self.explainers_w_args[idx] for idx in explainer_ids]

    def run(self, explainer_ids: Optional[Sequence[int]] = None) -> 'Experiment':
        explainers = self.get_explainers_by_ids(explainer_ids)
        runs = []

        for datum in self.data:
            print("DATUM: ", datum[0].shape)
            print("DATUM: ", datum[1].shape)
            for explainer in explainers:
                run = Run(
                    inputs=self.input_extractor(datum),
                    targets=self.target_extractor(datum),
                    explainer=explainer,
                    evaluator=self.evaluator,
                )
                run.execute()
                runs.append(run)

        self.runs = runs
        return self

    def visualize(self):
        visualizations = []
        for run in self.runs:
            run.visualize(task=self.task)

    def rank_by_metrics(self):
        pass
