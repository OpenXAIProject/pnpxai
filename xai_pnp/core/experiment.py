from torch import nn
from typing import Optional, List, Any
from dataclasses import dataclass

from time import time_ns
from xai_pnp.core._types import DataSource
from xai_pnp.explainers._explainer import Explainer

@dataclass
class Run:
    model: nn.Module
    inputs: DataSource
    outputs: Optional[DataSource] = None
    started_at: Optional[int] = None
    finished_at: Optional[int] = None


class Experiment:
    def __init__(
        self,
        model: nn.Module,
        explainer: Explainer
    ):
        self.model = model
        self.outputs = None
        self.explainer = explainer
        self.runs: List[Run] = []

    def _add_run(self, run: Run):
        self.runs.append(run)

    def run(
        self,
        data: DataSource,
        model: Optional[nn.Module] = None,
        *args: Any,
        **kwargs: Any
    ):
        if model is not None:
            self.model = model

        run = Run(
            model=self.model,
            inputs=data,
            started_at=time_ns()
        )

        run.outputs = self.explainer.run(self.model, data, *args, **kwargs)
        run.finished_at = time_ns()

        self._add_run(run)
