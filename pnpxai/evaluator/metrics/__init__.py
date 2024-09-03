from .mu_fidelity import MuFidelity
from .sensitivity import Sensitivity
from .complexity import Complexity
from .pixel_flipping import (
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
