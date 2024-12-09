from typing import Dict, Tuple, Optional
from torch import Tensor, nn
from captum.attr import LayerGradCam, LayerAttribution

from pnpxai.utils import format_into_tuple
from pnpxai.core.detector.types import Convolution
from .base import Explainer
from .utils import find_cam_target_layer
from .errors import NoCamTargetLayerAndNotTraceableError


class GradCam(Explainer):
    SUPPORTED_MODULES = [Convolution]

    def __init__(
        self,
        model: nn.Module,
        layer: Optional[nn.Module] = None,
        interpolate_mode: str = "bilinear",
    ) -> None:
        super().__init__(model)
        self._layer = layer
        self.interpolate_mode = interpolate_mode

    @property
    def layer(self):
        try:
            return self._layer or find_cam_target_layer(self.model)
        except:
            raise NoCamTargetLayerAndNotTraceableError(
                'You did not set cam target layer and',
                'it does not automatically determined.',
                'Please manually set the cam target layer by:',
                '`Explainer.set_target_layer(layer: nn.Module)`',
                'before attribute.'
            )

    def set_target_layer(self, layer: nn.Module):
        return self.set_kwargs(_layer=layer)

    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        forward_args, additional_forward_args = self._extract_forward_args(
            inputs)
        forward_args = format_into_tuple(forward_args)
        additional_forward_args = format_into_tuple(additional_forward_args)
        assert len(
            forward_args) == 1, 'GradCam for multiple inputs is not supported yet.'
        explainer = LayerGradCam(forward_func=self.model, layer=self.layer)
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

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        return {
            'interpolate_mode': (list, {'choices': ['bilinear', 'bicubic']}),
        }
