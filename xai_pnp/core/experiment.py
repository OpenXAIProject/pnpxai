from typing import Optional, List, Any
from dataclasses import dataclass

from time import time_ns
from xai_pnp.core._types import DataSource
from xai_pnp.explainers._explainer import Explainer


@dataclass
class Run:
    explainer: Explainer
    inputs: DataSource
    outputs: Optional[DataSource] = None
    started_at: Optional[int] = None
    finished_at: Optional[int] = None


class Experiment:
    def __init__(
        self,
        explainer: Explainer
    ):
        self.explainer = explainer
        self.runs: List[Run] = []

    def _add_run(self, run: Run):
        self.runs.append(run)

    def run(
        self,
        data: DataSource,
        *args: Any,
        **kwargs: Any
    ):
        run = Run(
            explainer=self.explainer,
            inputs=data,
            started_at=time_ns()
        )

        run.outputs = self.explainer.run(data, *args, **kwargs)
        run.finished_at = time_ns()

        self._add_run(run)

        return run

    def visualize(self):
        pass
