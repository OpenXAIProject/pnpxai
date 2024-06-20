from torch import Tensor
from torch.nn.modules import Module
from captum.attr import GuidedGradCam as CaptumGuidedGradCam

from .base import Explainer
from .utils import find_cam_target_layer


class GuidedGradCam(Explainer):
    def __init__(
            self,
            model: Module,
            interpolate_mode: str="nearest",
        ) -> None:
        super().__init__(model)
        self.interpolate_mode = interpolate_mode
        
    
    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        layer = find_cam_target_layer(self.model)
        explainer = CaptumGuidedGradCam(model=self.model, layer=layer)
        attrs = explainer.attribute(
            inputs=inputs,
            target=targets,
            interpolate_mode=self.interpolate_mode,
        )
        return attrs