import warnings
from _operator import add
from typing import Any, List, Sequence, Dict

import torch
from torch import Tensor, stack, nn
from captum._utils.typing import TargetType

from zennit.core import Composite
from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlus
from zennit.layer import Sum

from plotly import express as px
from plotly.graph_objects import Figure

from pnpxai.core._types import Model, DataSource
from pnpxai.detector import ModelArchitecture
from pnpxai.detector._core import NodeInfo
from pnpxai.explainers._explainer import Explainer

from .rules import MultiheadAttentionRule
from .utils import list_args_for_stack

class LRP(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = None #LRPCaptum(model)
            
    def attribute(
        self,
        inputs: DataSource,
        target: TargetType,
        attributor_type: Any = Gradient,
        canonizers: Any = [SequentialMergeBatchNorm()],
        composite_type: Any = EpsilonPlus,
        additional_composite_args: Dict = {},
        n_classes: int = 1000,
    ) -> List[Tensor]:
        # get model architecture to manipulate
        ma = ModelArchitecture(self.model)

        # find add functions from model and replace them to modules
        add_func_nodes = ma.find_node(
            lambda n: n.operator is add and all(isinstance(arg, NodeInfo) for arg in n.args),
            all = True
        )
        if add_func_nodes:
            warnings.warn(
                f"\n[Detector] Warning: {len(add_func_nodes)} add operations in function detected. Automatically changed to modules."
            )
        
        # replace
        for add_func_node in add_func_nodes:
            add_mod_node = ma.replace_node(
                add_func_node,
                NodeInfo.from_module(Sum()),
            )
            stack_node = ma.insert_node(
                NodeInfo.from_function(stack, dim=-1),
                add_mod_node,
                before = True,
            )
            _ = ma.insert_node(
                NodeInfo.from_function(list_args_for_stack),
                stack_node,
                before = True,
            )
        
        # find attention modules and assign rule for them
        atnt_node = ma.find_node(
            lambda n: isinstance(n.operator, nn.MultiheadAttention),
        )
        # TODO: better composite for vit (currently, expanse EpsilonPlus composite)
        if atnt_node:
            additional_composite_args["layer_map"] = [(nn.MultiheadAttention, MultiheadAttentionRule())]

        _model = ma.traced_model if add_func_nodes and atnt_node else self.model
        composite = composite_type(canonizers=canonizers, **additional_composite_args)

        if isinstance(target, int):
            targets = [target]
        elif torch.is_tensor(target):
            targets = target.tolist()
        else:
            raise Exception(f"Unsupported target type: {type(target)}")

        with attributor_type(model=_model, composite=composite) as attributor:
            _, relevance = attributor(inputs, torch.eye(n_classes)[targets])
        return relevance

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]