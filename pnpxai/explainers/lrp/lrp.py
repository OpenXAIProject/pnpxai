from typing import Any, List, Sequence
from torch import Tensor
from captum.attr import LRP as LRPCaptum
from captum._utils.typing import TargetType
from plotly import express as px
from plotly.graph_objects import Figure

from pnpxai.core._types import Model, DataSource
from pnpxai.explainers._explainer import Explainer


class LRP(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = LRPCaptum(model)

    def attribute(
        self,
        inputs: DataSource,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        verbose: bool = False
    ) -> List[Tensor]:
        attributions = self.method.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
            verbose=verbose
        )

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]
