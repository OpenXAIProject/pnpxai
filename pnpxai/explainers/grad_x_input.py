from typing import Callable, Optional, List, Tuple, Union, Sequence, Any

from torch import Tensor
from torch.nn.modules import Module
from captum.attr import InputXGradient as CaptumGradientXInput
from captum.attr import LayerGradientXActivation as CaptumLayerGradientXInput

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils import captum_wrap_model_input
from pnpxai.explainers.types import TargetLayerOrTupleOfTargetLayers


class GradientXInput(Explainer):
    """
    Grad X Input explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Avanti Shrikumar, Peyton Greenside, Anna Shcherbina, Anshul Kundaje. Not Just a Black Box: Learning Important Features Through Propagating Activation Differences.
    """

    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]
    SUPPORTED_DTYPES = [float, int]
    SUPPORTED_NDIMS = [2, 4]

    def __init__(
        self,
        model: Module,
        target_layer: TargetLayerOrTupleOfTargetLayers = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
    ) -> None:
        super().__init__(
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )
        self.target_layer = target_layer

    @property
    def _layer_explainer(self) -> CaptumLayerGradientXInput:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = [
            wrapped_model.input_maps[target_layer] if isinstance(target_layer, str)
            else target_layer for target_layer in self.target_layer
        ] if isinstance(self.target_layer, Sequence) else self.target_layer
        return CaptumLayerGradientXInput(
            forward_func=wrapped_model,
            target_layer=layers,
        )

    @property
    def _explainer(self) -> CaptumGradientXInput:
        return CaptumGradientXInput(forward_func=self.model)
    
    @property
    def explainer(self) -> Union[CaptumGradientXInput, CaptumLayerGradientXInput]:
        if self.target_layer is None:
            return self._explainer
        return self._layer_explainer

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
        attrs = self.explainer.attribute(
            inputs=forward_args,
            target=targets,
            additional_forward_args=additional_forward_args,
        )
        if isinstance(attrs, list):
            attrs = tuple(attrs)
        if isinstance(attrs, tuple) and len(attrs) == 1:
            attrs = attrs[0]
        return attrs
