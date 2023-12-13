import warnings
from time import time_ns
from typing import Optional, Any, Callable, List, Sequence
from plotly import express as px

from pnpxai.core._types import Task, DataSource
from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator import XaiEvaluator
from pnpxai.utils import class_to_string


class Run:
    def __init__(
        self,
        data: DataSource,
        input_extractor: Callable[[Any], Any],
        target_extractor: Callable[[Any], Any],
        explainer: ExplainerWArgs,
        evaluator: Optional[XaiEvaluator] = None,
    ):
        self.data = data
        self.explainer = explainer
        self.evaluator = evaluator

        self.input_extractor = input_extractor
        self.target_extractor = target_extractor

        n_data = len(self.data)
        self.explanations: List[DataSource] = [None for _ in range(n_data)]
        self.evaluations: List[DataSource] = [None for _ in range(n_data)]

        self.started_at: int
        self.finished_at: int

    def execute(self):
        self.started_at = time_ns()
        explainer_name = class_to_string(self.explainer.explainer)
        print(f"[Run] Explaining {explainer_name}")
        for i, datum in enumerate(self.data):
            try:
                inputs = self.input_extractor(datum)
                targets = self.target_extractor(datum)
                self.explanations[i] = self.explainer.attribute(
                    inputs=inputs,
                    targets=targets,
                )
            except NotImplementedError:
                warnings.warn(
                    f"\n[Run] Warning: {explainer_name} is not currently supported.")
            except Exception as e:
                warnings.warn(
                    f"\n[Run] Warning: Explaining {explainer_name} produced an error {e}.")

        print(f"[Run] Evaluating {explainer_name}")
        self.evaluate()

        self.finished_at = time_ns()

    def evaluate(self):
        if self.evaluator is None or self.explanations is None or not self.has_explanations:
            return None

        for i, (datum, explanation) in enumerate(zip(self.data, self.explanations)):
            if explanation is None:
                self.evaluations[i] = None
                continue

            inputs = self.input_extractor(datum)
            target = self.target_extractor(datum)

            self.evaluations[i] = self.evaluator(
                inputs, target, self.explainer, explanation
            )

    def visualize(self, task: Task):
        visualizations = []
        for datum, explanation in zip(self.data, self.explanations):
            if explanation is None:
                visualizations.append(None)
                continue

            inputs = self.input_extractor(datum)
            targets = self.target_extractor(datum)

            formatted = self.explainer.format_outputs_for_visualization(
                inputs=inputs,
                targets=targets,
                explanations=explanation,
                task=task
            )

            batch_visualizations = [
                px.imshow(explanation) for explanation in formatted
            ]
            visualizations.append(batch_visualizations)

        return visualizations

    def get_flattened_visualizations(self, task: Task):
        visualizations = self.visualize(task)
        flattened = []
        for datum, batch_visualizations in zip(self.data, visualizations):
            if batch_visualizations is None:
                batch_visualizations = [None for _ in range(len(datum))]

            flattened += batch_visualizations

        return flattened

    @property
    def flattened_evaluations(self) -> Optional[Sequence[Any]]:
        if self.evaluations is None:
            return None

        flattened = []
        for datum, batch_evaluations in zip(self.data, self.evaluations):
            if batch_evaluations is None:
                flattened.append([None for _ in range(len(datum))])
                continue

            metric_evaluations = []
            for idx, (metric, metric_vals) in enumerate(batch_evaluations.items()):
                if len(metric_evaluations) < len(metric_vals):
                    metric_evaluations += [
                        {} for _ in range(len(metric_vals) - len(metric_evaluations))
                    ]

                for idx, val in enumerate(metric_vals):
                    metric_evaluations[idx][metric] = val.item()

            flattened += metric_evaluations

        return flattened

    @property
    def flattened_weighted_evaluations(self):
        evaluations = self.flattened_evaluations
        if evaluations is None or self.evaluator is None:
            return None

        weighted_evaluations = [
            XaiEvaluator.weigh_metrics(evaluation)
            if evaluation is not None else None
            for evaluation in evaluations
        ]

        return weighted_evaluations
        

    def _has_vals(self, vals: Sequence[Optional[Any]]):
        return any(map(lambda x: x is not None, vals))

    @property
    def has_explanations(self):
        return self._has_vals(self.explanations)

    @property
    def has_evaluations(self):
        return self._has_vals(self.evaluations)

    @property
    def elapsed_time(self) -> Optional[int]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at
