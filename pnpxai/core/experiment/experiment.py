from typing import Optional, List, Any, Sequence, Dict, Callable
from dataclasses import dataclass
from plotly.graph_objects import Figure

from time import time_ns
from pnpxai.core.experiment.run import Run
from pnpxai.evaluator import XaiEvaluator
from pnpxai.core._types import DataSource
from pnpxai.explainers._explainer import ExplainerWArgs


class Experiment:
    def __init__(
        self,
        explainers_w_args: Sequence[ExplainerWArgs],
        evaluator: XaiEvaluator
    ):
        self.explainers_w_args = explainers_w_args
        self.evaluator = evaluator
        self.runs: List[Run] = []

    def _add_run(self, run: Run):
        self.runs.append(run)

    def run(
        self,
        inputs: DataSource,
        targets: DataSource,
    ) -> 'Experiment':
        for explainer_w_args in self.explainers_w_args:
            run = Run(
                inputs=inputs,
                targets=targets,
                explainer_w_args=explainer_w_args,
                evaluator=self.evaluator,
            )
            run.execute()
            
            self._add_run(run)

        return self

    def visualize(self, path: str = None) -> Sequence[List[Figure]]:
        return [
            run.explainer_w_args.explainer.format_outputs_for_visualization(
                run.inputs,
                run.explanation,
                run.explainer_w_args.args
            ) for run in self.runs
        ]
