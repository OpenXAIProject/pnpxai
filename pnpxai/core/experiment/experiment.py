from typing import Any, Callable, Optional, Sequence, Union
import time
import warnings

from torch import Tensor
from plotly import express as px
from plotly.graph_objects import Figure

from pnpxai.utils import class_to_string
from pnpxai.core.experiment.manager import ExperimentManager
from pnpxai.explainers import Explainer, ExplainerWArgs
from pnpxai.evaluator import EvaluationMetric
from pnpxai.core._types import DataSource, Model, Task


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
        metrics: Optional[Sequence[EvaluationMetric]] = None,
        task: Task = "image",
        input_extractor: Optional[Callable[[Any], Any]] = None,
        target_extractor: Optional[Callable[[Any], Any]] = None,
        input_visualizer: Optional[Callable[[Any], Any]] = None,
    ):
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
        self.task = task

    @property
    def all_explainers(self):
        return self.manager.all_explainers

    @property
    def all_metrics(self):
        return self.manager.all_metrics

    def get_current_explainers(self) -> Sequence[ExplainerWArgs]:
        return self.manager.get_explainers()[0]

    def get_current_metrics(self) -> Sequence[EvaluationMetric]:
        return self.manager.get_metrics()[0]

    def run(
        self,
        data_ids: Optional[Sequence[int]] = None,
        explainer_ids: Optional[Sequence[int]] = None,
        metrics_ids: Optional[Sequence[int]] = None,
    ) -> 'Experiment':
        self.manager.set_config(data_ids, explainer_ids, metrics_ids)
        explainers, explainer_ids = self.manager.get_explainers()

        for explainer, explainer_id in zip(explainers, explainer_ids):
            data, data_ids = self.manager.get_data_to_process_for_explainer(
                explainer_id)
            explanations = self._explain(data, explainer)
            self.manager.save_explanations(
                explanations, data, data_ids, explainer_id
            )

            metrics, metric_ids = self.manager.get_metrics()
            for metric, metric_id in zip(metrics, metric_ids):
                data, data_ids = self.manager.get_data_to_process_for_metric(
                    explainer_id, metric_id)
                explanations, data_ids = self.manager.get_valid_explanations(
                    explainer_id, data_ids)
                evaluations = self._evaluate(
                    data, explanations, explainer, metric)
                self.manager.save_evaluations(
                    evaluations, data, data_ids, explainer_id, metric_id)

        data, data_ids = self.manager.get_data_to_predict()
        outputs = [self.model(self.input_extractor(datum)) for datum in data]
        self.manager.save_outputs(outputs, data, data_ids)

        return self

    def _explain(self, data: DataSource, explainer: ExplainerWArgs):
        explanations = [None] * len(data)
        explainer_name = class_to_string(explainer.explainer)
        print(f'[Experiment] Explaining with {explainer_name}')
        for i, datum in enumerate(data):
            try:
                inputs = self.input_extractor(datum)
                targets = self.target_extractor(datum)
                explanations[i] = explainer.attribute(
                    inputs=inputs,
                    targets=targets,
                )
            except NotImplementedError:
                warnings.warn(
                    f"\n[Experiment] Warning: {explainer_name} is not currently supported.")
            except Exception as e:
                warnings.warn(
                    f"\n[Experiment] Warning: Explaining {explainer_name} produced an error: {e}.")
        return explanations

    def _evaluate(self, data: DataSource, explanations: DataSource, explainer: ExplainerWArgs, metric: EvaluationMetric):
        if explanations is None:
            return None
        started_at = time.time()
        metric_name = class_to_string(metric)
        explainer_name = class_to_string(explainer.explainer)
        print(f'[Experiment] Evaluating {metric_name} of {explainer_name}')
        evaluations = [None] * len(data)
        for i, (datum, explanation) in enumerate(zip(data, explanations)):
            if explanation is None:
                continue

            inputs = self.input_extractor(datum)
            targets = self.target_extractor(datum)
            try:
                evaluations[i] = metric(
                    self.model, explainer, inputs, targets, explanation
                )
            except Exception as e:
                warnings.warn(
                    f"\n[Experiment] Warning: Evaluating {metric_name} of {explainer_name} produced an error: {e}.")
        elaped_time = time.time() - started_at
        print(f'[Experiment] Computed {metric_name} in {elaped_time} sec')
        return evaluations

    def visualize_flat(self):
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
            explainer_visualizations = {idx: None for idx in experiment_data_ids}
            # Fill all valid visualizations
            for visualization, data_id in zip(flat_explainer_visualizations, data_ids):
                explainer_visualizations[data_id] = visualization
            visualizations.append(list(explainer_visualizations.values()))

        return visualizations

    def get_inputs_flattened(self) -> Sequence[Tensor]:
        data, _ = self.manager.get_data()
        data = [self.input_extractor(datum) for datum in data]
        return self.manager.flatten_if_batched(data, data)

    def get_targets_flattened(self) -> Sequence[Tensor]:
        data, _ = self.manager.get_data()
        targets = [self.target_extractor(datum)for datum in data]
        return self.manager.flatten_if_batched(targets, data)

    def get_outputs_flattened(self) -> Sequence[Tensor]:
        return self.manager.get_flat_outputs()

    def get_explanations_flattened(self) -> Sequence[Sequence[Tensor]]:
        _, explainer_ids = self.manager.get_explainers()
        return [
            self.manager.get_flat_explanations(explainer_id)
            for explainer_id in explainer_ids
        ]

    def get_evaluations_flattened(self) -> Sequence[Sequence[Sequence[Tensor]]]:
        _, explainer_ids = self.manager.get_explainers()
        _, metric_ids = self.manager.get_metrics()

        formatted = [[
            self.manager.get_flat_evaluations(explainer_id, metric_id)
            for metric_id in metric_ids
        ] for explainer_id in explainer_ids]

        return formatted

    def get_visualizations_flattened(self) -> Sequence[Sequence[Figure]]:
        return self.visualize_flat()

    @property
    def is_image_task(self):
        return self.task == 'image'
    
    @property
    def has_explanations(self):
        return self.manager.has_explanations
