from typing import Any, List
from torch import Tensor
from captum.attr import LayerIntegratedGradients as LayerIG

from xai_pnp.core._types import Model, DataSource
from xai_pnp.explainers._explainer import Explainer


class LayerIntegratedGradients(Explainer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def run(self, model: Model, data: DataSource, *args: Any, **kwargs: Any) -> List[Tensor]:
        lig = LayerIG(model, self.layer)

        attributions = []

        if type(data) is Tensor:
            data = [data]

        for datum in data:
            attributions.append(lig.attribute(datum, *args, **kwargs))

        return attributions
