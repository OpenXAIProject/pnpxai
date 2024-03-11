from typing import Type, Sequence

from pnpxai.evaluator._evaluator import EvaluationMetric
from pnpxai.explainers import Explainer, ExplainerWArgs
from pnpxai.core._types import Model

from pnpxai.core.experiment.experiment_explainer_defaults import EXPLAINER_AUTO_KWARGS
from pnpxai.core.experiment.experiment_metrics_defaults import EVALUATION_METRIC_AUTO_KWARGS

def init_explainers(model: Model, explainer_types: Sequence[Type[Explainer]]):
    return [
        ExplainerWArgs(
            explainer=explainer(model),
            kwargs=EXPLAINER_AUTO_KWARGS.get(explainer, None)
        )
        for explainer in explainer_types
    ]

    
def init_metrics(metric_types: Sequence[Type[EvaluationMetric]]):
    return [
        metric(**EVALUATION_METRIC_AUTO_KWARGS.get(metric, {}))
        for metric in metric_types
    ]