from typing import Optional, Literal

import torchvision.transforms.functional as TF
from torch.nn.modules import Module

from pnpxai.core.detector.types import Convolution
from pnpxai.core._types import Tensor
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
    elif mode == 'nearest_exact':
        return TF.InterpolationMode.NEAREST_EXACT
    elif mode == 'bicubic':
        return TF.InterpolationMode.BICUBIC
    return TF.InterpolationMode.BILINEAR


class FullGrad(ZennitExplainer):
    """
    FullGrad explainer.

    Supported Modules: `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        pooling_method (Optional[str]): The pooling mode used by the explainer. Available methods are: `"abssum"` (absolute sum) and `"possum"` (positive sum)
        interpolate_mode (Optional[str]): The interpolation mode used by the explainer. Available methods are: `"bilinear"`, `"nearest"`, `"nearest_exact"`, and `"bicubic"`
        n_classes (Optional[int]): The number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Suraj Srinivas, Francois Fleuret. Full-Gradient Representation for Neural Network Visualization.
    """

    SUPPORTED_MODULES = [Convolution]
    
    def __init__(
        self,
        model: Module,
        pooling_method: Literal['abssum', 'possum']='abssum',
        interpolate_mode: Literal['bilinear', 'nearest', 'nearest_exact', 'bicubic']='blinear',
        n_classes: Optional[int]=None,
    ):
        super().__init__(model=model, n_classes=n_classes)
        self.pooling_method = pooling_method
        self.interpolate_mode = interpolate_mode

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
            pooling_method=_format_pooling_method(self.pooling_method),
            interpolate_mode=_format_interpolate_mode(self.interpolate_mode),
        ) as attributor:
            attrs = attributor(inputs, self._format_targets(targets))
        return attrs

    def get_tunables(self):
        """
            Provides Tunable parameters for the optimizer

            Tunable parameters:
                `pooling_method` (str): Value can be selected of `"abssum"` and `"possum"`
                
                `interpolate_mode` (str): Value can be selected of `"bilinear"`, `"nearest"`, `"nearest_exact"`, and `"bicubic"`
        """
        return {
            'pooling_method': (list, {'choices': ['abssum', 'possum']}),
            'interpolate_mode': (list, {'choices': ['bilinear', 'nearest', 'nearest_exact', 'bicubic']}),
        }
