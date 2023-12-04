from typing import List, Type, Literal, Callable, Optional

from pnpxai.evaluator._evaluator import EvaluationMetric
from pnpxai.core.experiment.experiment import Experiment
from pnpxai.detector import ModelArchitectureDetector
from pnpxai.explainers import Explainer, ExplainerWArgs
from pnpxai.recommender.recommender import XaiRecommender, RecommenderOutput
from pnpxai.evaluator import XaiEvaluator
from pnpxai.core._types import DataSource, Model, Task, Question

from pnpxai.core.experiment.experiment_explainer_defaults import EXPLAINER_AUTO_KWARGS
from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_AUTO_KWARGS


class AutoExperiment(Experiment):
    def __init__(
        self,
        name: str,
        model: Model,
        data: DataSource,
        task: Literal["image", "tabular"] = "image",
        question: Literal["why", "how"] = "why",
        evaluator_enabled: bool = True,
        input_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
    ):
        recommender_output = self.recommend(model, question, task)

        explainers = self.__get_init_explainers(
            model, recommender_output.explainers)
        evaluator = self.__get_init_evaluator(recommender_output.evaluation_metrics)\
            if evaluator_enabled else None

        super().__init__(
            name=name,
            model=model,
            data=data,
            explainers=explainers,
            evaluator=evaluator,
            task=task,
            input_extractor=input_extractor,
            target_extractor=target_extractor,
        )

    @staticmethod
    def recommend(model: Model, question: Question, task: Task) -> RecommenderOutput:
        detector = ModelArchitectureDetector()
        model_arch = detector(model).architecture

        recommender = XaiRecommender()
        recommender_out = recommender(question, task, model_arch)

        return recommender_out

    def __get_init_explainers(self, model: Model, explainers: List[Type[Explainer]]) -> List[ExplainerWArgs]:
        return [
            ExplainerWArgs(
                explainer=explainer(model),
                kwargs=EXPLAINER_AUTO_KWARGS.get(explainer, None)
            )
            for explainer in explainers
        ]

    def __get_init_evaluator(self, metrics: List[Type[EvaluationMetric]]) -> XaiEvaluator:
        metrics = [
            metric(**EVALUATION_METRIC_AUTO_KWARGS.get(metric, {}))
            for metric in metrics
        ]
        return XaiEvaluator(metrics)

    def __repr__(self):
        return f"<AutoExperiment: {self.name}>"
