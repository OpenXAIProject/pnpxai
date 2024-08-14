from typing import Optional
from pnpxai.core.experiment.manager import ExperimentManager
from pnpxai.explainers.base import Explainer
from pnpxai.evaluator.metrics.base import Metric


class ExperimentObservableEvent:
    def __init__(
        self,
        manager: ExperimentManager,
        message: str,
        explainer: Optional[Explainer] = None,
        metric: Optional[Metric] = None
    ):
        self.message = message
        self.progress = self._compute_progress(manager, explainer, metric)

    def _compute_progress(
        self,
        manager: ExperimentManager,
        explainer: Optional[Explainer] = None,
        metric: Optional[Metric] = None
    ):
        explainers = manager.get_explainers()[0]
        metrics = manager.get_metrics()[0]

        progress = 0
        # Explainer has len(metrics) computations and its own
        n_explainer_states = len(metrics) + 1
        total_len = len(explainers) * (n_explainer_states)
        if explainer is not None:
            explainer_idx = explainers.index(explainer)
            # Offset number of computations before current and add current computation
            progress += explainer_idx * n_explainer_states + 1

        if metric is not None:
            metric_idx = metrics.index(metric)
            progress += metric_idx + 1

        return progress / total_len
