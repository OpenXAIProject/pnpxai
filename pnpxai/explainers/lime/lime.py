from typing import Dict
from captum.attr import Lime as LimeCaptum
from pnpxai.core._types import Model
from pnpxai.explainers._explainer import Explainer

class Lime(Explainer):
    def __init__(self, model: Model):
        super().__init__(
            source = LimeCaptum,
            model = model,
        )
    
    def get_default_additional_kwargs(self) -> Dict:
        return {
            "baselines": None,
            "additional_forward_args": None,
            "feature_mask": None, # felzenswalb
            "n_samples": 25,
            "perturbations_per_eval": 1,
            "return_input_shape": True,
            "show_progress": False,
        }