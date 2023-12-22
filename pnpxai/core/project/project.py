from typing import Optional, Dict, Callable, Sequence, Union

from pnpxai.core._types import DataSource, Model, Task, Question
from pnpxai.core.experiment import Experiment, AutoExperiment
from pnpxai.explainers._explainer import Explainer, ExplainerWArgs
from pnpxai.evaluator import EvaluationMetric
from pnpxai.visualizer.server import Server

EXPERIMENT_PREFIX = "experiment"


class Project():
    def __init__(self, name: str):
        self.name = name
        self.experiments: Dict[str, Experiment] = {}

        self._next_expr_id = 0
        self.__server = Server()
        self.__server.register(self)

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
        input_visualizer: Optional[Callable] = None,
        target_visualizer: Optional[Callable] = None,
    ) -> AutoExperiment:
        if name is None:
            name = self._generate_next_experiment_id()

        experiment = AutoExperiment(
            model=model,
            data=data,
            task=task,
            question=question,
            evaluator_enabled=evaluator_enabled,
            input_extractor=input_extractor,
            target_extractor=target_extractor,
            input_visualizer=input_visualizer,
            target_visualizer=target_visualizer
        )
        self.experiments[name] = experiment
        return experiment

    def create_experiment(
        self,
        model: Model,
        data: DataSource,
        name: Optional[str] = None,
        explainers: Optional[Sequence[Union[ExplainerWArgs, Explainer]]] = None,
        metrics: Optional[Sequence[EvaluationMetric]] = None,
        task: Task = "image",
        input_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        input_visualizer: Optional[Callable] = None,
        target_visualizer: Optional[Callable] = None,
    ) -> Experiment:
        if name is None:
            name = self._generate_next_experiment_id()

        experiment = Experiment(
            model=model,
            data=data,
            explainers=explainers,
            metrics=metrics,
            task=task,
            input_extractor=input_extractor,
            target_extractor=target_extractor,
            input_visualizer=input_visualizer,
            target_visualizer=target_visualizer
        )
        self.experiments[name] = experiment
        return experiment

    def __del__(self):
        self.__server.unregister(self)

    def get_server(self):
        return self.__server
