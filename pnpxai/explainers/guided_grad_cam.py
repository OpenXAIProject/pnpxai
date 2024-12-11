from typing import Dict, Any, Optional, Tuple
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import GuidedGradCam as CaptumGuidedGradCam

from pnpxai.core.detector.types import Convolution
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils import find_cam_target_layer
from pnpxai.utils import format_into_tuple
from .errors import NoCamTargetLayerAndNotTraceableError


class GuidedGradCam(Explainer):
    SUPPORTED_MODULES = [Convolution]

    def __init__(
        self,
        model: Module,
        layer: Optional[Module] = None,
        interpolate_mode: str = "nearest",
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

    def set_target_layer(self, layer: Module):
        return self.set_kwargs(_layer=layer)

    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        forward_args, additional_forward_args = self._extract_forward_args(
            inputs)
        forward_args = format_into_tuple(forward_args)
        additional_forward_args = format_into_tuple(additional_forward_args)
        assert len(
            forward_args) == 1, 'GuidedGradCam for multiple inputs is not supported yet.'
        explainer = CaptumGuidedGradCam(model=self.model, layer=self.layer)
        attrs = explainer.attribute(
            inputs=forward_args[0],
            target=targets,
            interpolate_mode=self.interpolate_mode,
        )
        return attrs

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        return {
            'interpolate_mode': (list, {'choices': ['nearest', 'area']}),
        }
