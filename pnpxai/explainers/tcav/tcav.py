from pnpxai.explainers._explainer import Explainer
from pnpxai.core._types import Model


class TCAV(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)

    def attribute(self, *args, **kwargs):
        raise NotImplementedError()
