from typing import Dict, List, Tuple, Callable, Sequence, Union, Optional

import _operator
import warnings

from torch import nn, fx, Tensor
from torch.nn.modules import Module

from zennit.attribution import Gradient
from zennit.core import Composite
from zennit.composites import (
    layer_map_base,
    LayerMapComposite,
    EpsilonGammaBox,
    EpsilonPlus,
    EpsilonAlpha2Beta1,
)
from zennit.types import Linear
from zennit.rules import Epsilon
from zennit.canonizers import SequentialMergeBatchNorm, Canonizer

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.explainers.attentions.module_converters import default_attention_converters
from pnpxai.explainers.attentions.rules import ConservativeAttentionPropagation
from pnpxai.explainers.zennit.attribution import Gradient, LayerGradient
from pnpxai.explainers.zennit.rules import LayerNormRule
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.zennit.layer import StackAndSum
from pnpxai.explainers.utils import captum_wrap_model_input
from pnpxai.evaluator.optimizer.utils import generate_param_key


class LRPBase(ZennitExplainer):
    def __init__(
        self,
        model: Module,
        zennit_composite: Composite,
        forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        layer: Optional[Union[Union[str, Module],
                              Sequence[Union[str, Module]]]] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes
        )
        self.zennit_composite = zennit_composite
        self.layer = layer

    def _layer_explainer(self, model: Union[Module, fx.GraphModule]) -> LayerGradient:
        wrapped_model = captum_wrap_model_input(model)
        stack = self.layer.copy() if isinstance(
            self.layer, Sequence) else [self.layer]
        layers = []
        while stack:
            layer = stack.pop(0)
            if isinstance(layer, str):
                layers.append(wrapped_model.input_maps[layer])
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
                        if module is layer:
                            layers.append(layer)
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
                        if any(anc is layer for anc in ancestors):
                            child_nodes.append(node)
                if not found:
                    last_child = self.model.get_submodule(
                        child_nodes[-1].target)
                    layers.append(last_child)
            elif isinstance(model, Module):
                layers.append(layer)
        if len(layers) == 1:
            layers = layers[0]
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
        if self.layer is None:
            return self._explainer(model)
        return self._layer_explainer(model)

    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor
    ) -> Union[Tensor, Tuple[Tensor]]:
        model = _replace_add_function_with_sum_module(self.model)
        forward_args, additional_forward_args = self._extract_forward_args(
            inputs)
        with self.explainer(model=model) as attributor:
            attrs = attributor.forward(
                forward_args,
                targets,
                additional_forward_args,
            )
        return attrs


class LRPUniformEpsilon(LRPBase):
    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]

    def __init__(
        self,
        model: Module,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = .25,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        layer: Optional[Union[str, Module,
                              Sequence[Union[str, Module]]]] = None,
        n_classes: Optional[int] = None
    ) -> None:
        self.epsilon = epsilon
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_uniform_epsilon_composite(
            epsilon, stabilizer, zennit_canonizers)
        super().__init__(
            model,
            zennit_composite,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            layer,
            n_classes
        )

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        return {
            'epsilon': (float, {"low": 1e-6, "high": 1, "log": True}),
        }


class LRPEpsilonGammaBox(LRPBase):
    SUPPORTED_MODULES = [Convolution]

    def __init__(
        self,
        model: Module,
        low: float = -3.,
        high: float = 3.,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        gamma: float = .25,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        layer: Optional[Union[str, Module,
                              Sequence[Union[str, Module]]]] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        self.low = low
        self.high = high
        self.epsilon = epsilon
        self.gamma = gamma
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_epsilon_gamma_box_composite(
            low, high, epsilon, gamma, stabilizer, zennit_canonizers)
        super().__init__(
            model,
            zennit_composite,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            layer,
            n_classes
        )

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        return {
            'epsilon': (float, {"low": 1e-6, "high": 1, "log": True}),
            'gamma': (float, {"low": 1e-6, "high": 1, "log": True}),
        }


class LRPEpsilonPlus(LRPBase):
    SUPPORTED_MODULES = [Convolution]

    def __init__(
        self,
        model: Module,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        layer: Optional[Union[str, Module,
                              Sequence[Union[str, Module]]]] = None,
        n_classes: Optional[int] = None
    ) -> None:
        self.epsilon = epsilon
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_epsilon_plus_composite(
            epsilon, stabilizer, zennit_canonizers)
        super().__init__(
            model,
            zennit_composite,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            layer,
            n_classes
        )

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        return {
            'epsilon': (float, {"low": 1e-6, "high": 1, "log": True}),
        }


class LRPEpsilonAlpha2Beta1(LRPBase):
    SUPPORTED_MODULES = [Convolution]

    def __init__(
        self,
        model: Module,
        epsilon: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        stabilizer: Union[float, Callable[[Tensor], Tensor]] = 1e-6,
        zennit_canonizers: Optional[List[Canonizer]] = None,
        forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        layer: Optional[Union[str, Module,
                              Sequence[Union[str, Module]]]] = None,
        n_classes: Optional[int] = None
    ) -> None:
        self.epsilon = epsilon
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers

        zennit_composite = _get_epsilon_alpha2_beta1_composite(
            epsilon, stabilizer, zennit_canonizers)
        super().__init__(
            model,
            zennit_composite,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            layer,
            n_classes
        )

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        return {
            'epsilon': (float, {"low": 1e-6, "high": 1, "log": True}),
        }


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


def _get_epsilon_gamma_box_composite(low, high, epsilon, gamma, stabilizer, zennit_canonizers):
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
    except:
        warnings.warn(
            "Your model cannot be traced by torch.fx.symbolic_trace.")
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
