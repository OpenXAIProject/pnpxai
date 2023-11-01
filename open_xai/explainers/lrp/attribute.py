from typing import Any, List, Sequence

from plotly import express as px
from plotly.graph_objects import Figure
import numpy as np

from typing import Dict

from torch import nn, Tensor
from captum._utils.typing import TargetType
from captum.attr._utils.lrp_rules import PropagationRule
from captum.attr._utils.custom_modules import Addition_Module

from open_xai.core._types import Model, DataSource
from open_xai.explainers._explainer import Explainer

from .lrp import _LRPBase
from .rules import epsilon_rule_factory, gamma_rule_factory


SUPPORTED_LINEAR_LAYERS = [
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    nn.Linear, nn.BatchNorm2d, Addition_Module,
]

class LRPBase(Explainer):
    def __init__(
            self,
            model: nn.Module,
            layer_map: Dict[nn.Module, PropagationRule],
        ):
        super().__init__(model)
        self.method = _LRPBase(model, layer_map)

    def run(
        self,
        data: DataSource,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        verbose: bool = False,
    ) -> List[Tensor]:
        attributions = []

        if type(data) is Tensor:
            data = [data]

        for datum in data:
            attributions.append(self.method.attribute(
                datum,
                target,
                additional_forward_args,
                return_convergence_delta= return_convergence_delta,
                verbose = verbose,
            ))

        return attributions

    def format_outputs_for_visualization(
            self,
            inputs: DataSource,
            outputs: DataSource,
            *args,
            **kwargs
        ) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]


class LRPEpsilon(LRPBase):
    def __init__(self, model: nn.Module, epsilon: float=1e-9):
        layer_map = {
            layer: epsilon_rule_factory(epsilon)
            for layer in SUPPORTED_LINEAR_LAYERS
        }
        super().__init__(model=model, layer_map=layer_map)
        self.epsilon = epsilon


class LRPGamma(LRPBase):
    def __init__(
            self,
            model: nn.Module,
            gamma: float = 0.25,
            set_bias_to_zero: bool = False
        ):
        layer_map = {
            layer: gamma_rule_factory(gamma, set_bias_to_zero)
            for layer in SUPPORTED_LINEAR_LAYERS
        }
        super().__init__(model=model, layer_map=layer_map)
        self.gamma = gamma