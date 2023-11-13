from time import time_ns
import warnings
from typing import Sequence, Optional, Callable, Any, List

from pnpxai.explainers import ExplainerWArgs
from pnpxai.evaluator import XaiEvaluator
from pnpxai.core._types import Model, Args, DataSource
from pnpxai.utils import class_to_string


class Run:
    def __init__(
        self,
        inputs: DataSource,
        targets: DataSource,
        explainer_w_args: ExplainerWArgs,
        evaluator: Optional[XaiEvaluator] = None,
    ):
        self.explainer_w_args = explainer_w_args
        self.evaluator = evaluator
        self.inputs = inputs
        self.targets = targets

        self.explanation: List[Any] = []
        self.evaluation: List[Any] = []

        self.started_at: int
        self.finished_at: int

    def execute(self):
        self.started_at = time_ns()
        explainer_method = class_to_string(self.explainer_w_args.explainer)
        print(f"[Run] Explaining {explainer_method}")
        for inputs, target in zip(self.inputs, self.targets):
            try:
                explainer = self.explainer_w_args.explainer
                explainer_args = self.explainer_w_args.args
                self.explanation.append(explainer.attribute(
                    inputs=inputs,
                    target=target,
                    *explainer_args.args,
                    **explainer_args.kwargs
                ))
            except NotImplementedError as e:
                warnings.warn(f"\n[Run] Warning: {repr(explainer)} is not currently supported.")
            
        print(f"[Run] Evaluating {explainer_method}")
        if self.evaluator is not None and len(self.explanation) > 0:
            inputs, target = next(iter(zip(self.inputs, self.targets)))

            explanation = self.explanation[0][:1]
            inputs = inputs[0][None, :]
            target = target[0]

            self.evaluation.append(self.evaluator(
                inputs, target, self.explainer_w_args, explanation
            ))

        self.finished_at = time_ns()

    @property
    def elapsed_time(self) -> Optional[int]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at
