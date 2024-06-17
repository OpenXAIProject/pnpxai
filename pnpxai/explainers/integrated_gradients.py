from typing import Callable, Optional, List, Tuple, Union, Sequence

import torch
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import IntegratedGradients as CaptumIntegratedGradients
from captum.attr import LayerIntegratedGradients as CaptumLayerIntegratedGradients

from .base import Explainer
from .utils import captum_wrap_model_input


class IntegratedGradients(Explainer):
    def __init__(
        self,
        model: Module,
        n_steps: int = 20,
        baseline_fn: Optional[Callable] = None,
        layer: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
    ) -> None:
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.layer = layer
        self.n_steps = n_steps
        self.baseline_fn = baseline_fn or torch.zeros_like


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
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        baselines = self.baseline_fn(forward_args)
        attrs = self.explainer.attribute(
            inputs=forward_args,
            baselines=baselines,
            target=targets,
            additional_forward_args=additional_forward_args,
            n_steps=self.n_steps,
        )
        return attrs