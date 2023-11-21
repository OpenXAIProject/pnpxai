from typing import Optional, List, Literal

from pnpxai.core.experiment import Experiment
from pnpxai.core._types import Model
from pnpxai.evaluator import XaiEvaluator

EXPERIMENT_PREFIX = "experiment"

class Project():
    def __init__(self, name: str):
        self.name = name
        self.experiments: List[Experiment] = []

        self._next_expr_id = 0
    
    def create_experiment(
            self,
            model: Model,
            task: Literal["image", "tabular"] = "image",
            name: Optional[str] = None,
            auto: bool = False,
            question: Literal["why", "how"] = "why", # if not auto, ignored.
            evaluator: Optional[XaiEvaluator] = None,
        ) -> Experiment:
        if not name:
            name = "_".join([EXPERIMENT_PREFIX, str(self._next_expr_id).zfill(2)])
        expr = Experiment(
            name = name,
            model = model,
            task = task,
            evaluator = evaluator,
        )
        if auto:
            expr.auto_add_explainers(question=question)
        self.experiments.append(expr)
        return expr