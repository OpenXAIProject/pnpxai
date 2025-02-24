from dataclasses import dataclass
from pnpxai.core._types import TensorOrTupleOfTensors
from pnpxai.explainers.base import Explainer
from pnpxai.evaluator.metrics.base import Metric


@dataclass
class ExperimentOutput:
    explainer: Explainer
    metric: Metric
    explanations: TensorOrTupleOfTensors
    evaluations: TensorOrTupleOfTensors
