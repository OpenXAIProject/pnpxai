from time import time_ns

from typing import Sequence, Optional, Callable, Any, List

from open_xai.explainers import ExplainerWArgs
from open_xai.evaluator import XaiEvaluator
from open_xai.core._types import Model, Args, DataSource


class Run:
    def __init__(
        self,
        inputs: DataSource,
        labels: DataSource,
        explainer_w_args: ExplainerWArgs,
        evaluator: Optional[XaiEvaluator] = None,
    ):
        self.explainer_w_args = explainer_w_args
        self.evaluator = evaluator
        self.inputs = inputs
        self.labels = labels

        self.explanation: List[Any] = None
        self.evaluation: List[Any] = None

        self.started_at: int
        self.finished_at: int

    def execute(self):
        self.started_at = time_ns()
        for batch in self.data:
            self.explanation.append(self.explainer_w_args.explainer.attribute(
                self.inputs,
                *self.explainer_w_args.args.args,
                **self.explainer_w_args.args.kwargs
            ))

        if self.evaluator is not None:
            inputs = self.inputs[0]
            labels = self.labels[0]
            self.evaluation.append(self.evaluator(
                inputs, labels, self.explainer_w_args, self.explanation[0]
            ))

        self.finished_at = time_ns()

    @property
    def elapsed_time(self) -> Optional[int]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at
