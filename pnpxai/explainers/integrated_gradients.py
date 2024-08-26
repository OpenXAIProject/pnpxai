from typing import Callable, Optional, Tuple, Union, Sequence, Dict

from torch import Tensor
from torch.nn.modules import Module
from captum.attr import IntegratedGradients as CaptumIntegratedGradients
from captum.attr import LayerIntegratedGradients as CaptumLayerIntegratedGradients

from pnpxai.core.detector.types import Linear, Convolution, Attention
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.explainers.utils.baselines import BaselineMethodOrFunction, BaselineFunction
from .base import Explainer
from .utils import captum_wrap_model_input


class IntegratedGradients(Explainer):
    SUPPORTED_MODULES = [Linear, Convolution, Attention]

    def __init__(
        self,
        model: Module,
        n_steps: int = 20,
        baseline_fn: Union[BaselineMethodOrFunction,
                           Tuple[BaselineMethodOrFunction]] = 'zeros',
        layer: Optional[Callable[[Tuple[Tensor]],
                                 Union[Tensor, Tuple[Tensor]]]] = None,
        forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
    ) -> None:
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.layer = layer
        self.n_steps = n_steps
        self.baseline_fn = baseline_fn

    @property
    def _layer_explainer(self) -> CaptumLayerIntegratedGradients:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = [
            wrapped_model.input_maps[layer] if isinstance(layer, str)
            else layer for layer in self.layer
        ] if isinstance(self.layer, Sequence) else self.layer
        return CaptumLayerIntegratedGradients(
            forward_func=wrapped_model,
            layer=layers,
        )

    @property
    def _explainer(self) -> CaptumIntegratedGradients:
        return CaptumIntegratedGradients(forward_func=self.model)

    @property
    def explainer(self) -> Union[CaptumIntegratedGradients, CaptumLayerIntegratedGradients]:
        if self.layer is None:
            return self._explainer
        return self._layer_explainer

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor
    ) -> Union[Tensor, Tuple[Tensor]]:
        forward_args, additional_forward_args = self._extract_forward_args(
            inputs)
        forward_args = format_into_tuple(forward_args)
        attrs = self.explainer.attribute(
            inputs=forward_args,
            baselines=self._get_baselines(forward_args),
            target=targets,
            additional_forward_args=additional_forward_args,
            n_steps=self.n_steps,
        )
        attrs = format_out_tuple_if_single(attrs)
        return attrs

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        return {
            'n_steps': (int, {'low': 10, 'high': 100, 'step': 10}),
            'baseline_fn': (BaselineFunction, {}),
        }
