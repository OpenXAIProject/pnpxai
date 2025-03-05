from typing import Optional, List, Union, Any, Callable
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import GuidedGradCam as CaptumGuidedGradCam

from pnpxai.core.detector.types import Convolution
from pnpxai.explainers.base import Explainer, Tunable
from pnpxai.explainers.types import TunableParameter
from pnpxai.explainers.utils import find_cam_target_layer
from pnpxai.explainers.errors import NoCamTargetLayerAndNotTraceableError


class GuidedGradCam(Explainer, Tunable):
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
    SUPPORTED_DTYPES = [float]
    SUPPORTED_NDIMS = [4]

    def __init__(
        self,
        model: Module,
        layer: Optional[Module] = None,
        interpolate_mode: str = "nearest",
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
    ) -> None:
        Explainer.__init__(
            self,
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )
        self._layer = layer
        self.interpolate_mode = TunableParameter(
            name='interpolate_mode',
            current_value=interpolate_mode,
            dtype=str,
            is_leaf=True,
            space={'choices': ['nearest', 'area']}
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.interpolate_mode])

    @property
    def layer(self):
        try:
            return self._layer or find_cam_target_layer(self.model)
        except Exception:
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
        forward_args, additional_forward_args = self.format_inputs(inputs)
        assert len(forward_args) == 1, (
            'GuidedGradCam for multiple inputs is not supported yet.',
        )
        explainer = CaptumGuidedGradCam(model=self.model, layer=self.layer)
        attrs = explainer.attribute(
            inputs=forward_args[0],
            target=targets,
            interpolate_mode=self.interpolate_mode.current_value,
        )
        return attrs
