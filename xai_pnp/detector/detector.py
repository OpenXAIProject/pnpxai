from xai_pnp.core._types import Model
from xai_pnp.explainers._explainer import Explainer
from xai_pnp.explainers import IntegratedGradients, RAP


class Detector:
    def __init__(self):
        pass

    def __call__(self, model: Model) -> Explainer:
        return RAP(model)
