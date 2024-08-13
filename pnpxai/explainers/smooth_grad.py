from typing import Tuple, Callable, Sequence, Union, Optional

from torch import Tensor
from torch.nn.modules import Module
from optuna.trial import Trial
from zennit.core import Composite

from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.explainers.zennit.attribution import SmoothGradient as SmoothGradAttributor
from pnpxai.explainers.zennit.attribution import LayerSmoothGradient as LayerSmoothGradAttributor
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.utils import captum_wrap_model_input, _format_to_tuple
from pnpxai.evaluator.optimizer.utils import generate_param_key


class SmoothGrad(ZennitExplainer):
    def __init__(
        self,
        model: Module,
        noise_level: float=.1,
        n_iter: int=20,
        square: bool=False,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        layer: Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]=None,
        n_classes: Optional[int]=None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes
        )
        self.noise_level = noise_level
        self.n_iter = n_iter
        self.layer = layer

    @property
    def _layer_attributor(self) -> LayerSmoothGradAttributor:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = [
            wrapped_model.input_maps[layer] if isinstance(layer, str)
            else layer for layer in self.layer
        ] if isinstance(self.layer, Sequence) else [self.layer]
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
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        with self.attributor() as attributor:
            grads = format_into_tuple(attributor.forward(
                forward_args,
                targets,
                additional_forward_args,
                return_squared=False,
            ))
        return format_out_tuple_if_single(grads)

    def suggest_tunables(self, trial: Trial, key: Optional[str]=None):
        return {
            'noise_level': trial.suggest_float(
                generate_param_key(key, 'noise_level'),
                low=.1, high=1., step=.1,
            ),
            'n_iter': trial.suggest_int(
                generate_param_key(key, 'n_iter'),
                low=10, high=100, step=10,
            ),
        }
