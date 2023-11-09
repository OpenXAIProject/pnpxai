from typing import Any, List, Sequence
from torch import Tensor
from captum.attr import GuidedGradCam as GuidedGradCamCaptum
from captum._utils.typing import TargetType
from plotly import express as px
from plotly.graph_objects import Figure

from open_xai.core._types import Model, DataSource
from open_xai.explainers._explainer import Explainer


class GuidedGradCam(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = GuidedGradCamCaptum(model)

    def attribute(
        self,
        data: DataSource,
        target: TargetType = None,
        additional_forward_args: Any = None,
        interpolate_mode: str = "nearest",
        attribute_to_layer_input: bool = False,
    ) -> List[Tensor]:
        attributions = []

        if type(data) is Tensor:
            data = [data]

        for datum in data:

            attributions.append(self.method.attribute(
                datum,
                target,
                additional_forward_args,
                interpolate_mode,
                attribute_to_layer_input,
            ))

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]
