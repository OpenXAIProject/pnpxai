from pnpxai.evaluator.metrics.base import Metric
from pnpxai.evaluator.metrics.mu_fidelity import MuFidelity
from pnpxai.evaluator.metrics.sensitivity import Sensitivity
from pnpxai.evaluator.metrics.complexity import Complexity
from pnpxai.evaluator.metrics.pixel_flipping import (
    PixelFlipping,
    MoRF,
    LeRF,
    AbPC,
)


PIXEL_FLIPPING_METRICS = [
    MoRF,
    LeRF,
    AbPC,
]

AVAILABLE_METRICS = PIXEL_FLIPPING_METRICS
