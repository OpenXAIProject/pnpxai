import warnings
from dataclasses import dataclass
from time import time_ns
from typing import Optional, Callable, Any
from functools import partial

from torch import Tensor

from pnpxai.core._types import Task
from pnpxai.explainers import ExplainerWArgs
from pnpxai.explainers.utils.post_process import postprocess_attr
from pnpxai.evaluator import XaiEvaluator
from pnpxai.core._types import DataSource


class Run:
    def __init__(
        self,
        inputs: DataSource,
        targets: DataSource,
        explainer: ExplainerWArgs,
        evaluator: Optional[XaiEvaluator] = None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.explainer = explainer
        self.evaluator = evaluator

        self.explanations: Any = None
        self.evaluations: Any = None

        self.started_at: int
        self.finished_at: int

    def execute(self):
        self.started_at = time_ns()
        print(f"[Run] Explaining {self.explainer.__class__.__name__}")
        try:
            self.explanations = self.explainer.attribute(
                inputs=self.inputs,
                targets=self.targets,
            )
        except NotImplementedError as e:
            warnings.warn(
                f"\n[Run] Warning: {repr(self.explainer)} is not currently supported.")

        print(f"[Run] Evaluating {self.explainer.__class__.__name__}")
        if self.evaluator is not None and self.explanations is not None and len(self.explanations) > 0:
            inputs, target, explanation = next(iter(zip(
                self.inputs, self.targets, self.explanations
            )))

            explanation = self.explanations[:1]
            inputs = inputs[None, :]

            self.evaluation = self.evaluator(
                inputs, target, self.explainer, explanation
            )

        self.finished_at = time_ns()

    def visualize(self, task: Task):
        explanations = self.explainer.format_outputs_for_visualization(
            inputs=self.inputs,
            targets=self.targets,
            explanations=self.explanations,
            task=task
        )

        return explanations

    @property
    def elapsed_time(self) -> Optional[int]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at
