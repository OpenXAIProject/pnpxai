from typing import Optional, List, Literal, Callable, Sequence, Union

from pnpxai.core._types import DataSource, Model, Task, Question
from pnpxai.core.experiment import Experiment, AutoExperiment
from pnpxai.explainers._explainer import Explainer, ExplainerWArgs
from pnpxai.evaluator import XaiEvaluator

EXPERIMENT_PREFIX = "experiment"


class Project():
    def __init__(self, name: str):
        self.name = name
        self.experiments: List[Experiment] = []

        self._next_expr_id = 0

    def _generate_next_experiment_id(self) -> str:
        idx = f"{EXPERIMENT_PREFIX}_{self._next_expr_id}"
        self._next_expr_id += 1
        return idx

    def create_auto_experiment(
        self,
        model: Model,
        data: DataSource,
        name: Optional[str] = None,
        task: Task = "image",
        question: Question = "why",
        evaluator_enabled: bool = True,
        input_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
    ) -> AutoExperiment:
        if name is None:
            name = self._generate_next_experiment_id()

        experiment = AutoExperiment(
            name=name,
            model=model,
            data=data,
            task=task,
            question=question,
            evaluator_enabled=evaluator_enabled,
            input_extractor=input_extractor,
            target_extractor=target_extractor
        )
        self.experiments.append(experiment)
        return experiment

    def create_experiment(
        self,
        model: Model,
        data: DataSource,
        name: Optional[str] = None,
        explainers: Optional[Sequence[Union[ExplainerWArgs, Explainer]]] = None,
        evaluator: Optional[XaiEvaluator] = None,
        task: Task = "image",
    ) -> Experiment:
        if name is None:
            name = self._generate_next_experiment_id()

        experiment = Experiment(
            name=name,
            model=model,
            data=data,
            explainers=explainers,
            evaluator=evaluator,
            task=task
        )
        self.experiments.append(experiment)
        return experiment
