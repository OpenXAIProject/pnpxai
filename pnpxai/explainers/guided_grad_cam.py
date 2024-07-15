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
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        assert len(forward_args) == 1, 'GuidedGradCam for multiple inputs is not supported yet.'
        layer = find_cam_target_layer(self.model)
        explainer = CaptumGuidedGradCam(model=self.model, layer=layer)
        attrs = explainer.attribute(
            inputs=forward_args[0],
            target=targets,
            interpolate_mode=self.interpolate_mode,
        )
        if isinstance(attrs, tuple) and len(attrs) == 1:
            attrs = attrs[0]
        return attrs