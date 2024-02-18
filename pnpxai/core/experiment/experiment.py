from typing import Any, Callable, Optional, Sequence, Union, List
import time
import warnings
import traceback

from torch import Tensor
import numpy as np
from plotly import express as px
from plotly.graph_objects import Figure

from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_REVERSE_SORT, EVALUATION_METRIC_SORT_PRIORITY
from pnpxai.core.experiment.observable import ExperimentObservableEvent
from pnpxai.utils import class_to_string, Observable
from pnpxai.messages import get_message
from pnpxai.core.experiment.manager import ExperimentManager
from pnpxai.explainers import Explainer, ExplainerWArgs
from pnpxai.evaluator import EvaluationMetric
from pnpxai.core._types import DataSource, Model, Task


def default_input_extractor(x):
    return x[0]


def default_target_extractor(x):
    return x[1]


class Experiment(Observable):
    """
    A class representing an experiment for model interpretability.

    Args:
        model (Model): The machine learning model to be interpreted.
        data (DataSource): The data used for the experiment.
        explainers (Sequence[Union[ExplainerWArgs, Explainer]]): Explainer objects or their arguments for interpreting the model.
        metrics (Optional[Sequence[EvaluationMetric]]): Evaluation metrics used to assess model interpretability.
        task (Task): The type of task the model is designed for (default: "image").
        input_extractor (Optional[Callable[[Any], Any]]): Function to extract inputs from data (default: None).
        target_extractor (Optional[Callable[[Any], Any]]): Function to extract targets from data (default: None).
        input_visualizer (Optional[Callable[[Any], Any]]): Function to visualize input data (default: None).
        target_visualizer (Optional[Callable[[Any], Any]]): Function to visualize target data (default: None).

    Attributes:
        all_explainers (Sequence[ExplainerWArgs]): All explainer objects used in the experiment.
        all_metrics (Sequence[EvaluationMetric]): All evaluation metrics used in the experiment.
        errors (Sequence[Error]): 
        is_image_task (bool): True if the task is an image-related task, False otherwise.
        has_explanations (bool): True if the experiment has explanations, False otherwise.
    """


    def __init__(
        self,
        model: Model,
        data: DataSource,
        explainers: Sequence[Union[ExplainerWArgs, Explainer]],
        metrics: Optional[Sequence[EvaluationMetric]] = None,
        task: Task = "image",
        input_extractor: Optional[Callable[[Any], Any]] = None,
        target_extractor: Optional[Callable[[Any], Any]] = None,
        input_visualizer: Optional[Callable[[Any], Any]] = None,
        target_visualizer: Optional[Callable[[Any], Any]] = None,
    ):
        super(Experiment, self).__init__()
        self.model = model
        self.manager = ExperimentManager(
            data=data,
            explainers=explainers,
            metrics=metrics or []
        )

        self.input_extractor = input_extractor \
            if input_extractor is not None \
            else default_input_extractor
        self.target_extractor = target_extractor \
            if target_extractor is not None \
            else default_target_extractor
        self.input_visualizer = input_visualizer
        self.target_visualizer = target_visualizer
        self.task = task
        self.reset_errors()

    def reset_errors(self):
        self._errors: List[BaseException] = []

    @property
    def errors(self):
        return self._errors

    @property
    def all_explainers(self) -> Sequence[ExplainerWArgs]:
        return self.manager.all_explainers

    @property
    def all_metrics(self) -> Sequence[EvaluationMetric]:
        return self.manager.all_metrics

    def get_current_explainers(self) -> List[ExplainerWArgs]:
        return self.manager.get_explainers()[0]

    def get_current_metrics(self) -> List[EvaluationMetric]:
        return self.manager.get_metrics()[0]

    def run(
        self,
        data_ids: Optional[Sequence[int]] = None,
        explainer_ids: Optional[Sequence[int]] = None,
        metrics_ids: Optional[Sequence[int]] = None,
    ) -> 'Experiment':
        """
        Run the experiment by processing data, generating explanations, evaluating with metrics, caching and retrieving the data.

        Args:
            data_ids (Optional[Sequence[int]]): A sequence of data IDs to specify the subset of data to process.
            explainer_ids (Optional[Sequence[int]]): A sequence of explainer IDs to specify the subset of explainers to use.
            metrics_ids (Optional[Sequence[int]]): A sequence of metric IDs to specify the subset of metrics to evaluate.

        Returns:
            The Experiment instance with updated results and state.

        This method orchestrates the experiment by configuring the manager, obtaining explainer and metric instances,
        processing data, generating explanations, and evaluating metrics. It then saves the results in the manager.

        Note: The input parameters allow for flexibility in specifying subsets of data, explainers, and metrics to process.
        If not provided, the method processes all available data, explainers, and metrics.
        """
        self.reset_errors()
        self.manager.set_config(data_ids, explainer_ids, metrics_ids)
        explainers, explainer_ids = self.manager.get_explainers()

        for explainer, explainer_id in zip(explainers, explainer_ids):
            explainer_name = class_to_string(explainer.explainer)
            data, data_ids = self.manager.get_data_to_process_for_explainer(
                explainer_id)
            explanations = self._explain(data, explainer)
            self.manager.save_explanations(
                explanations, data, data_ids, explainer_id
            )
            message = get_message(
                'experiment.event.explainer', explainer=explainer_name
            )
            print(f"[Experiment] {message}")
            self.fire(ExperimentObservableEvent(
                self.manager, message, explainer))

            metrics, metric_ids = self.manager.get_metrics()
            for metric, metric_id in zip(metrics, metric_ids):
                metric_name = class_to_string(metric)
                data, data_ids = self.manager.get_data_to_process_for_metric(
                    explainer_id, metric_id)
                explanations, data_ids = self.manager.get_valid_explanations(
                    explainer_id, data_ids)
                data, _ = self.manager.get_data(data_ids)
                evaluations = self._evaluate(
                    data, explanations, explainer, metric)
                self.manager.save_evaluations(
                    evaluations, data, data_ids, explainer_id, metric_id)

                message = get_message(
                    'experiment.event.explainer.metric', explainer=explainer_name, metric=metric_name)
                print(f"[Experiment] {message}")
                self.fire(ExperimentObservableEvent(
                    self.manager, message, explainer, metric))

        data, data_ids = self.manager.get_data_to_predict()
        outputs = [self.model(self.input_extractor(datum)) for datum in data]
        self.manager.save_outputs(outputs, data, data_ids)

        return self

    def _explain(self, data: DataSource, explainer: ExplainerWArgs):
        explanations = [None] * len(data)
        explainer_name = class_to_string(explainer.explainer)

        for i, datum in enumerate(data):
            try:
                inputs = self.input_extractor(datum)
                targets = self.target_extractor(datum)
                explanations[i] = explainer.attribute(
                    inputs=inputs,
                    targets=targets,
                )
            except NotImplementedError as error:
                warnings.warn(
                    f"\n[Experiment] {get_message('experiment.errors.explainer_unsupported', explainer=explainer_name)}")
                raise error
            except Exception as e:
                warnings.warn(
                    f"\n[Experiment] {get_message('experiment.errors.explanation', explainer=explainer_name, error=e)}")
                self._errors.append(e)
        
        return explanations

    def _evaluate(self, data: DataSource, explanations: DataSource, explainer: ExplainerWArgs, metric: EvaluationMetric):
        if explanations is None:
            return None
        started_at = time.time()
        metric_name = class_to_string(metric)
        explainer_name = class_to_string(explainer.explainer)

        evaluations = [None] * len(data)
        for i, (datum, explanation) in enumerate(zip(data, explanations)):
            if explanation is None:
                continue

            inputs = self.input_extractor(datum)
            targets = self.target_extractor(datum)
            try:
                # [GH] input args as kwargs to compute metric in an experiment
                evaluations[i] = metric(
                    model=self.model,
                    explainer_w_args=explainer,
                    inputs=inputs,
                    targets=targets,
                    attributions=explanation,
                )
            except Exception as e:
                warnings.warn(
                    f"\n[Experiment] {get_message('experiment.errors.evaluation', explainer=explainer_name, metric=metric_name, error=e)}")
                self._errors.append(e)
        elaped_time = time.time() - started_at
        print(f"[Experiment] {get_message('elapsed', task=metric_name, elapsed=elapsed_time)}")

        return evaluations

    def get_visualizations_flattened(self) -> Sequence[Sequence[Figure]]:
        """
        Generate flattened visualizations for each data point based on explanations.

        Returns:
            List of visualizations for each data point across explainers.

        This method retrieves valid explanations for each explainer, formats them for visualization,
        and generates visualizations using Plotly Express. The results are flattened based on the data points
        and returned as a list of lists of figures.
        """
        explainers, explainer_ids = self.manager.get_explainers()
        # Get all data ids
        experiment_data_ids = self.manager.get_data_ids()
        visualizations = []

        for explainer, explainer_id in zip(explainers, explainer_ids):
            # Get all valid explanations and data ids for this explainer
            explanations, data_ids = self.manager.get_valid_explanations(
                explainer_id)
            data = self.manager.get_data(data_ids)[0]
            explainer_visualizations = []
            # Visualize each valid explanataion
            for datum, explanation in zip(data, explanations):
                inputs = self.input_extractor(datum)
                targets = self.target_extractor(datum)
                formatted = explainer.format_outputs_for_visualization(
                    inputs=inputs,
                    targets=targets,
                    explanations=explanation,
                    task=self.task
                )

                if not self.manager.is_batched:
                    formatted = [formatted]
                formatted_visualizations = [
                    px.imshow(explanation, color_continuous_scale="Reds") for explanation in formatted
                ]
                if not self.manager.is_batched:
                    formatted_visualizations = formatted_visualizations[0]
                explainer_visualizations.append(formatted_visualizations)

            flat_explainer_visualizations = self.manager.flatten_if_batched(
                explainer_visualizations, data)
            # Set visualizaions of all data ids as None
            explainer_visualizations = {
                idx: None for idx in experiment_data_ids}
            # Fill all valid visualizations
            for visualization, data_id in zip(flat_explainer_visualizations, data_ids):
                explainer_visualizations[data_id] = visualization
            visualizations.append(list(explainer_visualizations.values()))

        return visualizations

    def get_inputs_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten last run input data.

        Returns:
            Flattened input data.

        This method retrieves input data using the input extractor and flattens it for further processing.
        """
        data, _ = self.manager.get_data()
        data = [self.input_extractor(datum) for datum in data]
        return self.manager.flatten_if_batched(data, data)

    def get_all_inputs_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten all input data.

        Returns:
            Flattened input data from all available data.

        This method retrieves input data from all available data points using the input extractor and flattens it.
        """
        data = self.manager.get_all_data()
        data = [self.input_extractor(datum) for datum in data]
        return self.manager.flatten_if_batched(data, data)

    def get_targets_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten target data.

        Returns:
            Flattened target data.

        This method retrieves target data using the target extractor and flattens it for further processing.
        """
        data, _ = self.manager.get_data()
        targets = [self.target_extractor(datum)for datum in data]
        return self.manager.flatten_if_batched(targets, data)

    def get_outputs_flattened(self) -> Sequence[Tensor]:
        """
        Retrieve and flatten model outputs.

        Returns:
            Flattened model outputs.

        This method retrieves flattened model outputs using the manager's get_flat_outputs method.
        """
        return self.manager.get_flat_outputs()

    def get_explanations_flattened(self) -> Sequence[Sequence[Tensor]]:
        """
        Retrieve and flatten explanations from all explainers.

        Returns:
            Flattened explanations from all explainers.

        This method retrieves flattened explanations for each explainer using the manager's get_flat_explanations method.
        """
        _, explainer_ids = self.manager.get_explainers()
        return [
            self.manager.get_flat_explanations(explainer_id)
            for explainer_id in explainer_ids
        ]

    def get_evaluations_flattened(self) -> Sequence[Sequence[Sequence[Tensor]]]:
        """
        Retrieve and flatten evaluations for all explainers and metrics.

        Returns:
            Flattened evaluations for all explainers and metrics.

        This method retrieves flattened evaluations for each explainer and metric using the manager's
        get_flat_evaluations method.
        """
        _, explainer_ids = self.manager.get_explainers()
        _, metric_ids = self.manager.get_metrics()

        formatted = [[
            self.manager.get_flat_evaluations(explainer_id, metric_id)
            for metric_id in metric_ids
        ] for explainer_id in explainer_ids]

        return formatted

    def get_explainers_ranks(self) -> Optional[Sequence[Sequence[int]]]:
        """
        Calculate and return rankings for explainers based on evaluations.

        Returns:
            Rankings of explainers. Returns None if rankings cannot be calculated.

        This method calculates rankings for explainers based on evaluations and metric scores. It considers
        metric priorities and sorting preferences to produce rankings.
        """
        evaluations = [[[
            data_evaluation.detach().cpu() if data_evaluation is not None else None
            for data_evaluation in metric_data
        ]for metric_data in explainer_data
        ]for explainer_data in self.get_evaluations_flattened()]
        # (explainers, metrics, data)
        evaluations = np.array(evaluations, dtype=float)
        evaluations = np.nan_to_num(evaluations, nan=-np.inf)
        if evaluations.ndim != 3:
            return None

        evaluations = evaluations.argsort(axis=-3).argsort(axis=-3) + 1
        n_explainers = evaluations.shape[0]
        metric_name_to_idx = {}

        for idx, metric in enumerate(self.get_current_metrics()):
            metric_name = class_to_string(metric)
            evaluations[:, idx, :] = evaluations[:, idx, :]
            if EVALUATION_METRIC_REVERSE_SORT.get(metric_name, False):
                evaluations[:, idx, :] = \
                    n_explainers - evaluations[:, idx, :] + 1

            metric_name_to_idx[metric_name] = idx
        # (explainers, data)
        scores: np.ndarray = evaluations.sum(axis=-2)

        for metric_name in EVALUATION_METRIC_SORT_PRIORITY:
            if metric_name not in metric_name_to_idx:
                continue

            idx = metric_name_to_idx[metric_name]
            scores = scores * n_explainers + evaluations[:, idx, :]

        return scores.argsort(axis=-2).argsort(axis=-2).tolist()

    @property
    def is_image_task(self):
        return self.task == 'image'

    @property
    def has_explanations(self):
        return self.manager.has_explanations
