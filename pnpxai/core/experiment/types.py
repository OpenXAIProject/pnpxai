from dataclasses import dataclass
from pnpxai.core._types import TensorOrTupleOfTensors


@dataclass
class ExperimentOutput:
    explanations: TensorOrTupleOfTensors
    evaluations: TensorOrTupleOfTensors
