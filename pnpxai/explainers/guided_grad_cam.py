from typing import Dict, Any, Optional, Tuple
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import GuidedGradCam as CaptumGuidedGradCam

from pnpxai.core.detector.types import Convolution
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils import find_cam_target_layer
from pnpxai.utils import format_into_tuple


class GuidedGradCam(Explainer):
    SUPPORTED_MODULES = [Convolution]

    def __init__(
        self,
        model: Module,
        interpolate_mode: str = "nearest",
    ) -> None:
        super().__init__(model)
        self.interpolate_mode = interpolate_mode

    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        forward_args, additional_forward_args = self._extract_forward_args(
            inputs)
        forward_args = format_into_tuple(forward_args)
        additional_forward_args = format_into_tuple(additional_forward_args)
        assert len(
            forward_args) == 1, 'GuidedGradCam for multiple inputs is not supported yet.'
        layer = find_cam_target_layer(self.model)
        explainer = CaptumGuidedGradCam(model=self.model, layer=layer)
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
