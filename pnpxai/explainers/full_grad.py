from typing import Optional, Literal, List, Union, Callable, Any

import torchvision.transforms.functional as TF
from torch.nn.modules import Module

from pnpxai.core.detector.types import Convolution
from pnpxai.core._types import Tensor
from pnpxai.explainers.base import Tunable
from pnpxai.explainers.types import TunableParameter
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.zennit.attribution import FullGradient as FullGradAttributor


def _format_pooling_method(method):
    if method == 'abssum':
        return lambda attrs: attrs.abs().sum(1)
    elif method == 'possum':
        return lambda attrs: attrs.clamp(min=0).sum(1)
    else:
        raise ValueError


def _format_interpolate_mode(mode):
    if mode == 'nearest':
        return TF.InterpolationMode.NEAREST
    elif mode == 'nearest-exact':
        return TF.InterpolationMode.NEAREST_EXACT
    elif mode == 'bicubic':
        return TF.InterpolationMode.BICUBIC
    return TF.InterpolationMode.BILINEAR


class FullGrad(ZennitExplainer, Tunable):
    """
    FullGrad explainer.

    Supported Modules: `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        pooling_method (Optional[str]): The pooling mode used by the explainer. Available methods are: `"abssum"` (absolute sum) and `"possum"` (positive sum)
        interpolate_mode (Optional[str]): The interpolation mode used by the explainer. Available methods are: `"bilinear"`, `"nearest"`, `"nearest-exact"`, and `"bicubic"`
        n_classes (Optional[int]): The number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Suraj Srinivas, Francois Fleuret. Full-Gradient Representation for Neural Network Visualization.
    """
    SUPPORTED_MODULES = [Convolution]
    SUPPORTED_DTYPES = [float]
    SUPPORTED_NDIMS = [4]

    def __init__(
        self,
        model: Module,
        pooling_method: Literal['abssum', 'possum'] = 'abssum',
        interpolate_mode: Literal['bilinear', 'bicubic', 'nearest', 'nearest-exact'] = 'blinear',
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        n_classes: Optional[int] = None,
    ):
        self.pooling_method = TunableParameter(
            name='pooling_method',
            current_value=pooling_method,
            dtype=str,
            is_leaf=True,
            space={'choices': ['abssum', 'possum']},
        )
        self.interpolate_mode = TunableParameter(
            name='interpolate_mode',
            current_value=interpolate_mode,
            dtype=str,
            is_leaf=True,
            space={'choices': ['bilinear', 'bicubic', 'nearest', 'nearest-exact']},
        )
        ZennitExplainer.__init__(
            self,
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
            n_classes,
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.pooling_method, self.interpolate_mode])

    def attribute(self, inputs: Tensor, targets: Tensor):
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            torch.Tensor: The result of the explanation.
        """
        with FullGradAttributor(
            model=self.model,
            pooling_method=_format_pooling_method(
                self.pooling_method.current_value),
            interpolate_mode=_format_interpolate_mode(
                self.interpolate_mode.current_value),
        ) as attributor:
            attrs = attributor(inputs, self._format_targets(targets))
        return attrs
