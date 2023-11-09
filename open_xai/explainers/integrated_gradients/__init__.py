from typing import Any, List, Sequence
import plotly.graph_objects as go
from torch import Tensor
from captum.attr import IntegratedGradients as IG
from captum._utils.typing import BaselineType, TargetType
from plotly import express as px
from plotly.graph_objects import Figure
import numpy as np

from open_xai.core._types import Model, DataSource
from open_xai.explainers._explainer import Explainer

from typing import Union, Literal


class IntegratedGradients(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = IG(model)

    def attribute(
        self,
        data: DataSource,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[True] = False
    ) -> List[Tensor]:
        attributions = []

        if type(data) is Tensor:
            data = [data]

        for datum in data:
            attributions.append(self.method.attribute(
                datum,
                baselines,
                target,
                additional_forward_args,
                n_steps,
                method,
                internal_batch_size,
                return_convergence_delta
            ))

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        attr_ig = []
        
        for attr, delta in outputs:
            attr = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            attr_ig.append(attr)

        return [px.imshow(attr) for attr in attr_ig]
