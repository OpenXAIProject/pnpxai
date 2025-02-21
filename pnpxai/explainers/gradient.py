from typing import Tuple, Callable, Sequence, Union, Optional, Any, List

from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.explainers.zennit.attribution import Gradient as GradientAttributor
from pnpxai.explainers.zennit.attribution import LayerGradient as LayerGradientAttributor
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.utils import captum_wrap_model_input
from pnpxai.explainers.types import TargetLayerOrTupleOfTargetLayers


class Gradient(ZennitExplainer):
    """
    Gradient explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained.
        n_classes (int): The number of classes.
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Gabriel Erion, Joseph D. Janizek, Pascal Sturmfels, Scott Lundberg, Su-In Lee. Improving performance of deep learning models with axiomatic attribution priors and expected gradients.
    """
    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]
    SUPPORTED_DTYPES = [float, int]
    SUPPORTED_NDIMS = [2, 4]

    def __init__(
        self,
        model: Module,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__(
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
            n_classes
        )
        self.target_layer = target_layer

    @property
    def _layer_attributor(self) -> LayerGradientAttributor:
        wrapped_model = captum_wrap_model_input(self._wrapped_model)
        layers = [
            wrapped_model.input_maps[target_layer] if isinstance(target_layer, str)
            else target_layer for target_layer in self.target_layer
        ] if isinstance(self.target_layer, Sequence) else [self.target_layer]
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
        if self.target_layer is None:
            return self._attributor
        return self._layer_attributor

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: The result of the explanation.
        """
        forward_args, additional_forward_args = self.format_inputs(inputs)
        attrs = self.attributor.forward(
            forward_args,
            targets,
            additional_forward_args,
        )
        return attrs
