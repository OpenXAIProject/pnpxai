from typing import Any, List
from torch import Tensor
from captum.attr import IntegratedGradients as IG

from xai_pnp.core._types import Model, DataSource
from xai_pnp.explainers._explainer import Explainer


class IntegratedGradients(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = IG(model)

    def run(self, data: DataSource, *args: Any, **kwargs: Any) -> List[Tensor]:
        attributions = []

        if type(data) is Tensor:
            data = [data]

        for datum in data:
            attributions.append(self.method.attribute(datum, *args, **kwargs))

        return attributions
