from typing import Any, List, Sequence, Optional
from torch import Tensor
from captum.attr import LayerGradCam
from captum._utils.typing import TargetType
from plotly import express as px
from plotly.graph_objects import Figure

from open_xai.core._types import Model, DataSource
from open_xai.explainers._explainer import Explainer
from open_xai.detector import ModelArchitecture

class GradCam(Explainer):
    def __init__(self, model: Model, layer: Optional[Model]=None):
        super().__init__(model)
        self.layer = layer if layer else self._locate_candidate_layer()
        self.method = LayerGradCam(forward_func=self.model, layer=self.layer)
    
    def _locate_candidate_layer(self):
        ma = ModelArchitecture.from_model(self.model)
        candidate = ma.find_cam_target_module() # (name, module)
        assert candidate, "GradCam applicable layer not found."
        print(f"GradCam target layer not given. Automatically set it as '{candidate[0]}'")
        module = self.model
        for s in candidate[0].split("."):
            module = getattr(module, s)
        return module

    def run(
        self,
        data: DataSource,
        target: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
    ) -> List[Tensor]:
        attributions = []

        if type(data) is Tensor:
            data = [data]

        for datum in data:
            attributions.append(self.method.attribute(
                datum,
                target,
                additional_forward_args,
                attribute_to_layer_input = attribute_to_layer_input,
                relu_attributions = relu_attributions,
            ))

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]
