from typing import Optional, List, Any, Sequence, Dict
from dataclasses import dataclass
from plotly.graph_objects import Figure

from time import time_ns
from open_xai.core._types import DataSource
from open_xai.explainers._explainer import Explainer


@dataclass
class ExplainerConfig:
    args: Sequence[Any]
    kwargs: Dict[str, Any]


@dataclass
class Run:
    explainer: Explainer
    explainer_config: ExplainerConfig
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
            explainer_config=ExplainerConfig(args=args, kwargs=kwargs),
            inputs=data,
            started_at=time_ns(),
        )

        run.outputs = self.explainer.run(
            data, *run.explainer_config.args, **run.explainer_config.kwargs
        )
        run.finished_at = time_ns()

        self._add_run(run)

        return run

    def visualize(self) -> Sequence[List[Figure]]:
        return [
            run.explainer.format_outputs_for_visualization(
                run.inputs,
                run.outputs,
                *run.explainer_config.args,
                **run.explainer_config.kwargs
            ) for run in self.runs
        ]
