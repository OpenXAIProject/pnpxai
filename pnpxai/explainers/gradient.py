from typing import Tuple, Callable, Sequence, Union, Optional

from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from .zennit.attribution import Gradient as GradientAttributor
from .zennit.attribution import LayerGradient as LayerGradientAttributor
from .zennit.base import ZennitExplainer
from .utils import captum_wrap_model_input


class Gradient(ZennitExplainer):
    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]

    def __init__(
        self,
        model: Module,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        layer: Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]=None,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes
        )
        self.layer = layer

    @property
    def _layer_attributor(self) -> LayerGradientAttributor:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = [
            wrapped_model.input_maps[layer] if isinstance(layer, str)
            else layer for layer in self.layer
        ] if isinstance(self.layer, Sequence) else [self.layer]
        if len(layers) == 1:
            layers = layers[0]
        return LayerGradientAttributor(
            model=wrapped_model,
            layer=layers,
        )

    @property
    def _attributor(self) -> GradientAttributor:
        return GradientAttributor(self.model)
    
    @property
    def attributor(self) -> Union[GradientAttributor, LayerGradientAttributor]:
        if self.layer is None:
            return self._attributor
        return self._layer_attributor

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor
    ) -> Union[Tensor, Tuple[Tensor]]:
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        attrs = self.attributor.forward(
            forward_args,
            targets,
            additional_forward_args,
        )
        return attrs