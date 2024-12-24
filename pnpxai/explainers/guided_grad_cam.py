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
    """
    GuidedGradCam explainer.

    Supported Modules: `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        interpolate_mode (Optional[str]): The interpolation mode used by the explainer. Available methods are: `"bilinear"` and `"bicubic"`
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    """

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
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            torch.Tensor: The result of the explanation.
        """
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
        """
        Provides Tunable parameters for the optimizer

        Tunable parameters:
            `interpolate_mode` (str): Value can be selected of `"bilinear"` and `"bicubic"`
        """
        return {
            'interpolate_mode': (list, {'choices': ['nearest', 'area']}),
        }
