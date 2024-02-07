from typing import List, Type, Literal, Callable, Optional, Sequence

from pnpxai.evaluator._evaluator import EvaluationMetric
from pnpxai.core.experiment.experiment import Experiment
from pnpxai.detector import ModelArchitectureDetector
from pnpxai.explainers import Explainer, ExplainerWArgs
from pnpxai.recommender.recommender import XaiRecommender, RecommenderOutput
from pnpxai.core._types import DataSource, Model, Task, Question

from pnpxai.core.experiment.experiment_explainer_defaults import EXPLAINER_AUTO_KWARGS
from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_AUTO_KWARGS


class AutoExperiment(Experiment):
    """
    An extension of Experiment class with automatic explainers and evaluation metrics recommendation.

    Parameters:
        model (Model): The machine learning model to be analyzed.
        data (DataSource): The data source used for the experiment.
        task (Literal["image", "tabular"], optional): The task type, either "image" or "tabular". Defaults to "image".
        question (Literal["why", "how"], optional): The type of question the experiment aims to answer, either "why" or "how". Defaults to "why".
        evaluator_enabled (bool, optional): Whether to enable the evaluator. Defaults to True.
        input_extractor (Optional[Callable], optional): Custom function to extract input features. Defaults to None.
        target_extractor (Optional[Callable], optional): Custom function to extract target labels. Defaults to None.
        input_visualizer (Optional[Callable], optional): Custom function for visualizing input features. Defaults to None.
        target_visualizer (Optional[Callable], optional): Custom function for visualizing target labels. Defaults to None.
    """
    def __init__(
        self,
        model: Model,
        data: DataSource,
        task: Literal["image", "tabular"] = "image",
        question: Literal["why", "how"] = "why",
        evaluator_enabled: bool = True,
        input_extractor: Optional[Callable] = None,
        target_extractor: Optional[Callable] = None,
        input_visualizer: Optional[Callable] = None,
        target_visualizer: Optional[Callable] = None,
    ):
        recommender_output = self.recommend(model, question, task)

        explainers = self.__get_init_explainers(
            model, recommender_output.explainers)
        metrics = self.__get_init_evaluator(recommender_output.evaluation_metrics)\
            if evaluator_enabled else None

        super().__init__(
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

    @staticmethod
    def recommend(model: Model, question: Question, task: Task) -> RecommenderOutput:
        """
        Recommend explainers and metrics based on the model architecture.

        Parameters:
            model (Model): The machine learning model to recommend explainers for.
            question (Question): The type of question the experiment aims to answer.
            task (Task): The type of task the model is designed for.

        Returns:
            RecommenderOutput: Output containing recommended explainers and metrics.
        """
        detector = ModelArchitectureDetector()
        model_arch = detector(model).architecture

        recommender = XaiRecommender()
        recommender_out = recommender(question, task, model_arch)

        return recommender_out

    def __get_init_explainers(self, model: Model, explainers: List[Type[Explainer]]) -> List[ExplainerWArgs]:
        """
        Initialize and configure explainer instances with default arguments.

        Parameters:
            model (Model): The machine learning model for which explainers are initialized.
            explainers (List[Type[Explainer]]): List of explainer classes to be initialized.

        Returns:
            List[ExplainerWArgs]: List of initialized ExplainerWArgs instances.
        """
        return [
            ExplainerWArgs(
                explainer=explainer(model),
                kwargs=EXPLAINER_AUTO_KWARGS.get(explainer, None)
            )
            for explainer in explainers
        ]

    def __get_init_evaluator(self, metrics: List[Type[EvaluationMetric]]) -> Sequence[EvaluationMetric]:
        """
        Initialize and configure evaluator metrics with default arguments.

        Parameters:
            metrics (List[Type[EvaluationMetric]]): List of metric classes to be initialized.

        Returns:
            Sequence[EvaluationMetric]: Sequence of initialized EvaluationMetric instances.
        """
        return [
            metric(**EVALUATION_METRIC_AUTO_KWARGS.get(metric, {}))
            for metric in metrics
        ]
