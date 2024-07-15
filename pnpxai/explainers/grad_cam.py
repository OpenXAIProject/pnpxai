from torch import Tensor, nn
from captum.attr import LayerGradCam, LayerAttribution

from .base import Explainer
from .utils import find_cam_target_layer


class GradCam(Explainer):
    def __init__(
            self,
            model: nn.Module,
            interpolate_mode: str="bilinear",
        ) -> None:
        super().__init__(model)
        self.interpolate_mode = interpolate_mode
        

    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        assert len(forward_args) == 1, 'GradCam for multiple inputs is not supported yet.'
        layer = find_cam_target_layer(self.model)
        explainer = LayerGradCam(forward_func=self.model, layer=layer)
        attrs = explainer.attribute(
            forward_args[0],
            target=targets,
            additional_forward_args=additional_forward_args,
            attr_dim_summation=True,
        )
        upsampled = LayerAttribution.interpolate(
            layer_attribution=attrs,
            interpolate_dims=forward_args[0].shape[2:],
            interpolate_mode=self.interpolate_mode,
        )
        return upsampled