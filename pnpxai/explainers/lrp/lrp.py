from typing import Dict

from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import LayerMapComposite

from pnpxai.core._types import Model
from pnpxai.explainers._explainer import Explainer
from .captum_like import CaptumLikeZennit

class LRP(Explainer):
    def __init__(self, model: Model):
        super().__init__(
            source = CaptumLikeZennit,
            model = model,
        )
    
    def get_default_additional_kwargs(self) -> Dict:
        return {
            "attributor_type": Gradient,
            "canonizers": [SequentialMergeBatchNorm()],
            "composite_type": LayerMapComposite,
            "additional_composite_args": {},
            "n_classes": 1000,
        }