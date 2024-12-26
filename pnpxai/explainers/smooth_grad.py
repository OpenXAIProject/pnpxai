from typing import Tuple, Callable, Sequence, Union, Optional, Dict

from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.explainers.zennit.attribution import SmoothGradient as SmoothGradAttributor
from pnpxai.explainers.zennit.attribution import (
    LayerSmoothGradient as LayerSmoothGradAttributor,
)
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.utils import captum_wrap_model_input


class SmoothGrad(ZennitExplainer):
    """
    SmoothGrad explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        noise_level (float): The added noise level.
        n_iter (int): The Number of iterations the algorithm makes
        layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg. SmoothGrad: removing noise by adding noise.
    """

    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]

    def __init__(
        self,
        model: Module,
        noise_level: float = 0.1,
        n_iter: int = 20,
        forward_arg_extractor: Optional[
            Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]
        ] = None,
        additional_forward_arg_extractor: Optional[
            Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]
        ] = None,
        layer: Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__(
            model, forward_arg_extractor, additional_forward_arg_extractor, n_classes
        )
        self.noise_level = noise_level
        self.n_iter = n_iter
        self.layer = layer

    @property
    def _layer_attributor(self) -> LayerSmoothGradAttributor:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = (
            [
                wrapped_model.input_maps[layer] if isinstance(layer, str) else layer
                for layer in self.layer
            ]
            if isinstance(self.layer, Sequence)
            else [self.layer]
        )
        if len(layers) == 1:
            layers = layers[0]
        return LayerSmoothGradAttributor(
            model=wrapped_model,
            layer=layers,
            noise_level=self.noise_level,
            n_iter=self.n_iter,
        )

    @property
    def _attributor(self) -> SmoothGradAttributor:
        return SmoothGradAttributor(
            model=self.model,
            noise_level=self.noise_level,
            n_iter=self.n_iter,
        )

    def attributor(self) -> Union[SmoothGradAttributor, LayerSmoothGradAttributor]:
        if self.layer is None:
            return self._attributor
        return self._layer_attributor

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: The result of the explanation.
        """
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        with self.attributor() as attributor:
            grads = format_into_tuple(
                attributor.forward(
                    forward_args,
                    targets,
                    additional_forward_args,
                    return_squared=False,
                )
            )
        return format_out_tuple_if_single(grads)

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        """
        Provides Tunable parameters for the optimizer

        Tunable parameters:
            `noise_level` (float): Value can be selected in the range of `range(0, 0.95, 0.05)`

            `n_iter` (int): Value can be selected in the range of `range(10, 100, 10)`
        """
        return {
            "noise_level": (float, {"low": 0.05, "high": 0.95, "step": 0.05}),
            "n_iter": (int, {"low": 10, "high": 100, "step": 10}),
        }
