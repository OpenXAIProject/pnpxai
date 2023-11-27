import warnings
from dataclasses import dataclass
from time import time_ns
from typing import Optional, Callable, Any
from functools import partial

from torch import Tensor

from pnpxai.explainers import Explainer
from pnpxai.explainers.utils.post_process import postprocess_attr
from pnpxai.evaluator import XaiEvaluator
from pnpxai.core._types import DataSource

class Run:
    def __init__(
        self,
        inputs: DataSource,
        targets: DataSource,
        explainer: Explainer,
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
                    inputs = self.inputs,
                    targets = self.targets,
            )
        except NotImplementedError as e:
            warnings.warn(f"\n[Run] Warning: {repr(self.explainer)} is not currently supported.")
            
        print(f"[Run] Evaluating {self.explainer.__class__.__name__}")
        if self.evaluator is not None and len(self.explanation) > 0:
            inputs, target = next(iter(zip(self.inputs, self.targets)))

            explanation = self.explanation[0][:1]
            inputs = inputs[0][None, :]
            target = target[0]

            self.evaluation.append(self.evaluator(
                inputs, target, self.explainer, explanation
            ))

        self.finished_at = time_ns()
    
    def to_heatmap_data(
            self,
            input_process: Callable = lambda input: input,
            target_process: Callable = lambda target: target,
            expl_process: Optional[Callable] = None,
        ):
        data = []
        expl_process = (
            partial(postprocess_attr, sign="absolute")
            if expl_process is None else expl_process
        )
        for input, target, expl in zip(
            self.inputs,
            self.targets,
            self.explanations
        ):
            data.append(HeatmapData(
                input = input_process(input),
                target = target_process(target),
                explanation = expl_process(expl),
            ))
        return data
        
    @property
    def elapsed_time(self) -> Optional[int]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

@dataclass
class HeatmapData:
    input: Tensor
    target: int
    explanation: Tensor
