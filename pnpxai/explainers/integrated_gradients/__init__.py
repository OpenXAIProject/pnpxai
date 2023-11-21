from typing import Dict
from captum.attr import IntegratedGradients as IG

from pnpxai.core._types import Model, DataSource
from pnpxai.explainers._explainer import Explainer


class IntegratedGradients(Explainer):
    def __init__(self, model: Model):
        super().__init__(
            source = IG,
            model = model,
        )

    def get_default_additional_kwargs(self) -> Dict:
        return {
            "baselines": None,
            "additional_forward_args": None,
            "n_steps": 50,
            "method": "gausslegendre",
            "internal_batch_size": None,
        }
