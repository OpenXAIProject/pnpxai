from open_xai.core._types import Model
from open_xai.explainers._explainer import Explainer
from open_xai.explainers import IntegratedGradients, RAP


class Detector:
    def __init__(self):
        pass

    def __call__(self, model: Model) -> Explainer:
        return RAP(model)
