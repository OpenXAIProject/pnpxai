import warnings
from dataclasses import dataclass
from time import time_ns
from typing import Optional, Any, Callable, List
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
            except NotImplementedError as e:
                warnings.warn(
                    f"\n[Run] Warning: {explainer_name} is not currently supported.")

        print(f"[Run] Evaluating {explainer_name}")
        if self.evaluator is not None and self.explanations is not None and self.has_explanations:
            datum = None
            explanation = None

            for datum, explanation in zip(self.data, self.explanations):
                if explanation is not None:
                    break

            inputs = self.input_extractor(datum)#[:1]
            target = self.target_extractor(datum)#[:1]
            # explanation = explanation#[:1]

            self.evaluations = self.evaluator(
                inputs, target, self.explainer, explanation
            )

        self.finished_at = time_ns()

    def visualize(self, task: Task):
        visualizations = []
        for datum, explanation in zip(self.data, self.explanations):
            if explanation is None:
                visualizations.append([None for _ in range(len(self.datum))])
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

    @property
    def has_explanations(self):
        return any(map(lambda x: x is not None, self.explanations))

    @property
    def elapsed_time(self) -> Optional[int]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at
