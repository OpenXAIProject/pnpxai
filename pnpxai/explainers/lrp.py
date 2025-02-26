from typing import Dict, List, Tuple, Callable, Sequence, Union, Optional, Any

import _operator
import warnings

from torch import nn, fx, Tensor
from torch.nn.modules import Module

from zennit.core import Composite
from zennit.composites import (
    layer_map_base,
    LayerMapComposite,
    EpsilonGammaBox,
    EpsilonPlus,
    EpsilonAlpha2Beta1,
)
from zennit.rules import Epsilon
from zennit.canonizers import SequentialMergeBatchNorm, Canonizer

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.explainers.attentions.module_converters import default_attention_converters
from pnpxai.explainers.attentions.rules import ConservativeAttentionPropagation
from pnpxai.explainers.base import Tunable
from pnpxai.explainers.zennit.attribution import Gradient, LayerGradient
from pnpxai.explainers.zennit.rules import LayerNormRule
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.zennit.layer import StackAndSum
from pnpxai.explainers.utils import ModelWrapperForLayerAttribution
from pnpxai.explainers.types import (
    TargetLayerOrTupleOfTargetLayers,
    TunableParameter,
)
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


class LRPBase(ZennitExplainer):
    """
    Base class for `LRPUniformEpsilon`, `LRPEpsilonGammaBox`, `LRPEpsilonPlus`, and `LRPEpsilonAlpha2Beta1` explainers.

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        zennit_composite (Composite): The Composite object applies canonizers and register hooks to modules. One Composite instance may only be applied to a single module at a time.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).        
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Bach S., Binder A., Montavon G., Klauschen F., M¨uller K.-R., and Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation.
    """
    def __init__(
        self,
        model: Module,
        zennit_composite: Composite,
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
        self.zennit_composite = zennit_composite
        self.target_layer = target_layer

    def _layer_explainer(
        self,
        model: Union[Module, fx.GraphModule],
    ) -> LayerGradient:
        wrapped_model = ModelWrapperForLayerAttribution(self._wrapped_model)
        stack = list(format_into_tuple(self.target_layer))
        # stack = self.target_layer.copy() if isinstance(
        #     self.target_layer, Sequence) else [self.target_layer]
        layers = ()
        while stack:
            target_layer = stack.pop(0)
            if isinstance(target_layer, str):
                layers += (wrapped_model.input_maps[target_layer],)
                continue
            if isinstance(model, fx.GraphModule):
                child_nodes = []
                found = False
                for node in model.graph.nodes:
                    if node.op == "call_module":
                        try:
                            module = self.model.get_submodule(node.target)
                        except AttributeError:
                            continue
                        if module is target_layer:
                            layers += (target_layer,)
                            found = True
                            break
                        path_to_node = node.target.split(".")[:-1]
                        if len(path_to_node) == 0:
                            continue
                        ancestors = [
                            self.model.get_submodule(
                                ".".join(path_to_node[:i+1]))
                            for i in range(len(path_to_node))
                        ]
                        if any(anc is target_layer for anc in ancestors):
                            child_nodes.append(node)
                if not found:
                    last_child = self.model.get_submodule(
                        child_nodes[-1].target)
                    layers += (last_child,)
            elif isinstance(model, Module):
                layers += (target_layer,)
        layers = format_out_tuple_if_single(layers)
        return LayerGradient(
            model=wrapped_model,
            layer=layers,
            composite=self.zennit_composite,
        )

    def _explainer(self, model) -> Gradient:
        return Gradient(
            model=model,
            composite=self.zennit_composite
        )

    def explainer(self, model) -> Union[Gradient, LayerGradient]:
        if self.target_layer is None:
            return self._explainer(model)
        return self._layer_explainer(model)

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
            torch.Tensor: The result of the explanation.
        """
        model = _replace_add_function_with_sum_module(self.model)
        forward_args, additional_forward_args = self.format_inputs(
            inputs)
        with self.explainer(model=model) as attributor:
            attrs = attributor.forward(
                forward_args,
                targets,
                additional_forward_args,
            )
        return attrs


class LRPUniformEpsilon(LRPBase, Tunable):
    """
    LRPUniformEpsilon explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        epsilon (Union[float, Callable[[Tensor], Tensor]]): The epsilon value.
        stabilizer (Union[float, Callable[[Tensor], Tensor]]): The stabilizer value
        zennit_canonizers (Optional[List[Canonizer]]): An optional list of canonizers. Canonizers modify modules temporarily such that certain attribution rules can properly be applied.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer
    """

    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]
    SUPPORTED_DTYPES = [float, int]
    SUPPORTED_NDIMS = [2, 4]

    def __init__(
        self,
        model: Module,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = .25,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        n_classes: Optional[int] = None
    ) -> None:
        self.epsilon = TunableParameter(
            name='epsilon',
            current_value=epsilon,
            dtype=float,
            is_leaf=True,
            space={"low": 1e-6, "high": 1, "log": True},
        )
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_uniform_epsilon_composite(
            self.epsilon.current_value, stabilizer, zennit_canonizers)
        LRPBase.__init__(
            self,
            model,
            zennit_composite,
            target_input_keys,
            additional_input_keys,
            output_modifier,
            target_layer,
            n_classes
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.epsilon])


class LRPEpsilonGammaBox(LRPBase, Tunable):
    """
    LRPEpsilonGammaBox explainer.

    Supported Modules: `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        low (float): The lowest possible value for computing gamma box
        high (float): The highest possible value for computing gamma box
        gamma (float): The gamma value for computing gamma box
        epsilon (Union[float, Callable[[Tensor], Tensor]]): The epsilon value.
        stabilizer (Union[float, Callable[[Tensor], Tensor]]): The stabilizer value
        zennit_canonizers (Optional[List[Canonizer]]): An optional list of canonizers. Canonizers modify modules temporarily such that certain attribution rules can properly be applied.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer
    """

    SUPPORTED_MODULES = [Convolution]
    SUPPORTED_DTYPES = [float]
    SUPPORTED_NDIMS = [4]

    def __init__(
        self,
        model: Module,
        low: float = -3.,
        high: float = 3.,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        gamma: float = .25,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        self.low = low
        self.high = high
        self.epsilon = TunableParameter(
            name='epsilon',
            current_value=epsilon,
            dtype=float,
            is_leaf=True,
            space={"low": 1e-6, "high": 1, "log": True},
        )
        self.gamma = TunableParameter(
            name='gamma',
            current_value=gamma,
            dtype=float,
            is_leaf=True,
            space={"low": 1e-6, "high": 1, "log": True},
        )
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_epsilon_gamma_box_composite(
            low, high, self.epsilon.current_value, self.gamma.current_value,
            stabilizer, zennit_canonizers,
        )
        LRPBase.__init__(
            self,
            model,
            zennit_composite,
            target_input_keys,
            additional_input_keys,
            output_modifier,
            target_layer,
            n_classes
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.epsilon, self.gamma])


class LRPEpsilonPlus(LRPBase, Tunable):
    """
    LRPEpsilonPlus explainer.

    Supported Modules: `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        epsilon (Union[float, Callable[[Tensor], Tensor]]): The epsilon value.
        stabilizer (Union[float, Callable[[Tensor], Tensor]]): The stabilizer value
        zennit_canonizers (Optional[List[Canonizer]]): An optional list of canonizers. Canonizers modify modules temporarily such that certain attribution rules can properly be applied.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer
    """

    SUPPORTED_MODULES = [Convolution]
    SUPPORTED_DTYPES = [float]
    SUPPORTED_NDIMS = [4]

    def __init__(
        self,
        model: Module,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        n_classes: Optional[int] = None
    ) -> None:
        self.epsilon = TunableParameter(
            name='epsilon',
            current_value=epsilon,
            dtype=float,
            is_leaf=True,
            space={"low": 1e-6, "high": 1, "log": True},
        )
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_epsilon_plus_composite(
            self.epsilon.current_value, stabilizer, zennit_canonizers)
        LRPBase.__init__(
            self,
            model,
            zennit_composite,
            target_input_keys,
            additional_input_keys,
            output_modifier,
            target_layer,
            n_classes
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.epsilon])


class LRPEpsilonAlpha2Beta1(LRPBase, Tunable):
    """
    LRPEpsilonAlpha2Beta1 explainer.

    Supported Modules: `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        epsilon (Union[float, Callable[[Tensor], Tensor]]): The epsilon value.
        stabilizer (Union[float, Callable[[Tensor], Tensor]]): The stabilizer value
        zennit_canonizers (Optional[List[Canonizer]]): An optional list of canonizers. Canonizers modify modules temporarily such that certain attribution rules can properly be applied.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained
        n_classes (Optional[int]): Number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer
    """

    SUPPORTED_MODULES = [Convolution]
    SUPPORTED_DTYPES = [float]
    SUPPORTED_NDIMS = [4]
    TUNABLES = {
        'epsilon': (float, {"low": 1e-6, "high": 1, "log": True}),
    }

    def __init__(
        self,
        model: Module,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        n_classes: Optional[int] = None
    ) -> None:
        self.epsilon = TunableParameter(
            name='epsilon',
            current_value=epsilon,
            dtype=float,
            is_leaf=True,
            space={"low": 1e-6, "high": 1, "log": True},
        )
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_epsilon_alpha2_beta1_composite(
            self.epsilon.current_value, stabilizer, zennit_canonizers)
        LRPBase.__init__(
            self,
            model,
            zennit_composite,
            target_input_keys,
            additional_input_keys,
            output_modifier,
            target_layer,
            n_classes
        )
        Tunable.__init__(self)
        self.register_tunable_params([self.epsilon])


def _get_uniform_epsilon_composite(epsilon, stabilizer, zennit_canonizers):
    zennit_canonizers = zennit_canonizers or []
    canonizers = canonizers_base() + default_attention_converters + zennit_canonizers
    layer_map = (
        [(Linear, Epsilon(epsilon=epsilon))]
        + transformer_layer_map(stabilizer=stabilizer)
        + layer_map_base(stabilizer=stabilizer)
    )
    composite = LayerMapComposite(layer_map=layer_map, canonizers=canonizers)
    return composite


def _get_epsilon_gamma_box_composite(
    low,
    high,
    epsilon,
    gamma,
    stabilizer,
    zennit_canonizers,
):
    zennit_canonizers = zennit_canonizers or []
    canonizers = canonizers_base() + default_attention_converters + zennit_canonizers
    composite = EpsilonGammaBox(
        low=low,
        high=high,
        epsilon=epsilon,
        gamma=gamma,
        stabilizer=stabilizer,
        layer_map=transformer_layer_map(stabilizer=stabilizer),
        canonizers=canonizers,
    )
    return composite


def _get_epsilon_plus_composite(epsilon, stabilizer, zennit_canonizers):
    zennit_canonizers = zennit_canonizers or []
    canonizers = canonizers_base() + default_attention_converters + zennit_canonizers
    composite = EpsilonPlus(
        epsilon=epsilon,
        stabilizer=stabilizer,
        layer_map=transformer_layer_map(stabilizer=stabilizer),
        canonizers=canonizers,
    )
    return composite


def _get_epsilon_alpha2_beta1_composite(epsilon, stabilizer, zennit_canonizers):
    zennit_canonizers = zennit_canonizers or []
    canonizers = canonizers_base() + default_attention_converters + zennit_canonizers
    composite = EpsilonAlpha2Beta1(
        epsilon=epsilon,
        stabilizer=stabilizer,
        layer_map=transformer_layer_map(stabilizer=stabilizer),
        canonizers=canonizers,
    )
    return composite


def transformer_layer_map(stabilizer: float = 1e-6):
    layer_map = [
        (nn.MultiheadAttention, ConservativeAttentionPropagation(stabilizer=stabilizer)),
        (nn.LayerNorm, LayerNormRule(stabilizer=stabilizer)),
    ]
    return layer_map


def canonizers_base():
    return [SequentialMergeBatchNorm()]


def _replace_add_function_with_sum_module(model: Module) -> fx.GraphModule:
    try:
        traced_model = fx.symbolic_trace(model)
    except Exception:
        warnings.warn(
            "Your model may not be traced by torch.fx.symbolic_trace.")
        return model
    treated = False
    for node in traced_model.graph.nodes:
        if node.target is _operator.add:
            treated = True
            with traced_model.graph.inserting_after(node):
                traced_model.add_submodule(
                    f"_replaced_{node.name}", StackAndSum())
                replacement = traced_model.graph.call_module(
                    f"_replaced_{node.name}", args=node.args)
                node.replace_all_uses_with(replacement)
                traced_model.graph.erase_node(node)
    if not treated:
        return model
    # ensure changes
    traced_model.graph.lint()
    traced_model.recompile()
    return traced_model
