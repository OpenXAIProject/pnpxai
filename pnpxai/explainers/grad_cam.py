from typing import Dict, Tuple, Optional
from torch import Tensor, nn
from captum.attr import LayerGradCam, LayerAttribution

from pnpxai.utils import format_into_tuple
from pnpxai.core.detector.types import Convolution
from .base import Explainer
from .utils import find_cam_target_layer
from .errors import NoCamTargetLayerAndNotTraceableError


class GradCam(Explainer):
    """
    GradCAM explainer.

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
        self, model: nn.Module, interpolate_mode: str = "bilinear", **kwargs
    ) -> None:
        super().__init__(model, **kwargs)
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
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            torch.Tensor: The result of the explanation.
        """
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        forward_args = format_into_tuple(forward_args)
        additional_forward_args = format_into_tuple(additional_forward_args)

        assert (
            len(forward_args) == 1
        ), "GradCam for multiple inputs is not supported yet."
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

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        """
        Provides Tunable parameters for the optimizer

        Tunable parameters:
            `interpolate_mode` (str): Value can be selected of `"bilinear"` and `"bicubic"`
        """
        return {
            "interpolate_mode": (list, {"choices": ["bilinear", "bicubic"]}),
        }
