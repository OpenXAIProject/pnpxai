from typing import Any, Sequence
from torch import Tensor
from captum.attr import IntegratedGradients as IntegratedGradientsCaptum
from captum._utils.typing import BaselineType, TargetType
from plotly import express as px
from plotly.graph_objects import Figure
import numpy as np

from pnpxai.core._types import Model, DataSource
from pnpxai.explainers._explainer import Explainer

from typing import Union, Literal


class IntegratedGradients(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = IntegratedGradientsCaptum(model)

    def attribute(
        self,
        inputs: DataSource,
        target: TargetType = None,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[True] = False
    ) -> Tensor:
        attributions = self.method.attribute(
            inputs,
            baselines,
            target,
            additional_forward_args,
            n_steps,
            method,
            internal_batch_size,
            return_convergence_delta
        )

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        attr_ig = []

        for attr, delta in outputs:
            attr = np.transpose(
                attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            attr_ig.append(attr)

        return [px.imshow(attr) for attr in attr_ig]
