from typing import Dict, Any, Optional, Tuple
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import GuidedGradCam as CaptumGuidedGradCam

from pnpxai.core.detector.types import Convolution
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils import find_cam_target_layer
from pnpxai.utils import format_into_tuple


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
        interpolate_mode: str = "nearest",
    ) -> None:
        super().__init__(model)
        self.interpolate_mode = interpolate_mode

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
        layer = find_cam_target_layer(self.model)
        explainer = CaptumGuidedGradCam(model=self.model, layer=layer)
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
