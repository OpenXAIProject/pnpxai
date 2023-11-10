from typing import Any, List, Sequence, Optional
from torch import Tensor, nn
from captum.attr import GuidedGradCam as GuidedGradCamCaptum
from captum._utils.typing import TargetType
from plotly import express as px
from plotly.graph_objects import Figure

from open_xai.core._types import Model, DataSource
from open_xai.explainers._explainer import Explainer


class GuidedGradCam(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)

        layer = self._find_last_conv_layer(self.model.modules())
        self.method = GuidedGradCamCaptum(model, layer)

    def _find_last_conv_layer(self, modules: Sequence[nn.Module]) -> Optional[nn.Conv2d]:
        last_conv = None
        for module in modules:
            if isinstance(module, nn.Conv2d):
                last_conv = module

            submodules = list(module.children())
            if len(submodules) > 0:
                last_conv = self._find_last_conv_layer(submodules)
        
        return last_conv

    def attribute(
        self,
        inputs: DataSource,
        target: TargetType = None,
        additional_forward_args: Any = None,
        interpolate_mode: str = "nearest",
        attribute_to_layer_input: bool = False,
    ) -> List[Tensor]:
        attributions=self.method.attribute(
            inputs,
            target,
            additional_forward_args,
            interpolate_mode,
            attribute_to_layer_input,
        )

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]
