from typing import Tuple, Callable, Sequence, Union, Optional, Any, List

from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.explainers.base import Tunable
from pnpxai.explainers.types import TunableParameter, TargetLayerOrTupleOfTargetLayers
from pnpxai.explainers.zennit.attribution import SmoothGradient as SmoothGradAttributor
from pnpxai.explainers.zennit.attribution import (
    LayerSmoothGradient as LayerSmoothGradAttributor,
)
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.utils import ModelWrapperForLayerAttribution


class SmoothGrad(ZennitExplainer, Tunable):
    """
    SmoothGrad explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        noise_level (float): The added noise level.
        n_iter (int): The Number of iterations the algorithm makes
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, Martin Wattenberg. SmoothGrad: removing noise by adding noise.
    """

    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]
    SUPPORTED_DTYPES = [float, int]
    SUPPORTED_NDIMS = [2, 4]

    def __init__(
        self,
        model: Module,
        noise_level: float = 0.1,
        n_iter: int = 20,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        ZennitExplainer.__init__(
            self,
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
            n_classes,
        )
        self.noise_level = TunableParameter(
            name='noise_level',
            current_value=noise_level,
            dtype=float,
            is_leaf=True,
            space={"low": 0.05, "high": 0.95, "step": 0.05},
        )
        self.n_iter = TunableParameter(
            name='n_iter',
            current_value=n_iter,
            dtype=int,
            is_leaf=True,
            space={"low": 10, "high": 100, "step": 10},
        )
        self.target_layer = target_layer
        Tunable.__init__(self)
        self.register_tunable_params([self.noise_level, self.n_iter])

    @property
    def _layer_attributor(self) -> LayerSmoothGradAttributor:
        wrapped_model = ModelWrapperForLayerAttribution(self._wrapped_model)
        layers = [
            wrapped_model.input_maps[target_layer] if isinstance(target_layer, str) else target_layer
            for target_layer in format_into_tuple(self.target_layer)
        ]
        layers = format_out_tuple_if_single(layers)
        return LayerSmoothGradAttributor(
            model=wrapped_model,
            layer=layers,
            noise_level=self.noise_level.current_value,
            n_iter=self.n_iter.current_value,
        )

    @property
    def _attributor(self) -> SmoothGradAttributor:
        return SmoothGradAttributor(
            model=self.model,
            noise_level=self.noise_level.current_value,
            n_iter=self.n_iter.current_value,
        )

    def attributor(self) -> Union[SmoothGradAttributor, LayerSmoothGradAttributor]:
        if self.target_layer is None:
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
        forward_args, additional_forward_args = self.format_inputs(inputs)
        with self.attributor() as attributor:
            grads = format_into_tuple(
                attributor.forward(
                    format_out_tuple_if_single(forward_args),
                    targets,
                    additional_forward_args,
                    return_squared=False,
                )
            )
        return format_out_tuple_if_single(grads)
