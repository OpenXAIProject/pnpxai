from typing import Callable, Optional, Tuple, Union, Sequence, List, Any

from torch import Tensor
from torch.nn.modules import Module
from captum.attr import IntegratedGradients as CaptumIntegratedGradients
from captum.attr import LayerIntegratedGradients as CaptumLayerIntegratedGradients

from pnpxai.core.detector.types import Linear, Convolution, Attention
from pnpxai.utils import format_out_tuple_if_single
from pnpxai.utils import (
    format_multimodal_supporting_input,
    run_multimodal_supporting_util_fn,
)
from pnpxai.explainers.base import Explainer, Tunable
from pnpxai.explainers.types import (
    TargetLayerOrTupleOfTargetLayers,
    TunableParameter,
)
from pnpxai.explainers.utils import captum_wrap_model_input
from pnpxai.explainers.utils.types import BaselineFunctionOrTupleOfBaselineFunctions
from pnpxai.explainers.utils.baselines import ZeroBaselineFunction


class IntegratedGradients(Explainer, Tunable):
    """
    IntegratedGradients explainer.

    Supported Modules: `Linear`, `Convolution`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        baseline_fn (Union[BaselineMethodOrFunction, Tuple[BaselineMethodOrFunction]]): The baseline function, accepting the attribution input, and returning the baseline accordingly.
        n_steps (int): The Number of steps the algorithm makes
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Mukund Sundararajan, Ankur Taly, Qiqi Yan. Axiomatic Attribution for Deep Networks.
    """

    SUPPORTED_MODULES = [Linear, Convolution, Attention]
    SUPPORTED_DTYPES = [float, int]
    SUPPORTED_NDIMS = [2, 4]

    def __init__(
        self,
        model: Module,
        n_steps: int = 20,
        baseline_fn: Optional[BaselineFunctionOrTupleOfBaselineFunctions] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
    ) -> None:
        self.target_layer = target_layer
        self.n_steps = TunableParameter(
            name='n_steps',
            current_value=n_steps,
            dtype=int,
            is_leaf=True,
            space={'low': 10, 'high': 100, 'step': 10},
        )
        baseline_fn = baseline_fn or ZeroBaselineFunction()
        self.baseline_fn = format_multimodal_supporting_input(
            baseline_fn,
            format=TunableParameter,
            input_key='current_value',
            name='baseline_fn',
            dtype=str,
            is_leaf=False,
        )
        Explainer.__init__(
            self,
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.n_steps, self.baseline_fn])

    @property
    def _layer_explainer(self) -> CaptumLayerIntegratedGradients:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = [
            wrapped_model.input_maps[target_layer] if isinstance(target_layer, str)
            else target_layer for target_layer in self.target_layer
        ] if isinstance(self.target_layer, Sequence) else self.target_layer
        return CaptumLayerIntegratedGradients(
            forward_func=wrapped_model,
            layer=layers,
        )

    @property
    def _explainer(self) -> CaptumIntegratedGradients:
        return CaptumIntegratedGradients(forward_func=self.model)

    @property
    def explainer(self) -> Union[CaptumIntegratedGradients, CaptumLayerIntegratedGradients]:
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
        baselines = run_multimodal_supporting_util_fn(forward_args, self.baseline_fn)
        attrs = self.explainer.attribute(
            inputs=forward_args,
            baselines=baselines,
            target=targets,
            additional_forward_args=additional_forward_args,
            n_steps=self.n_steps.current_value,
        )
        if isinstance(attrs, tuple):
            attrs = format_out_tuple_if_single(attrs)
        return attrs
