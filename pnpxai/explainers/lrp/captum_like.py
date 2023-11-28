import warnings
from _operator import add
from typing import Any, List, Dict

import torch
from torch import nn
from captum._utils.typing import TargetType

from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import LayerMapComposite, layer_map_base
from zennit.layer import Sum

from pnpxai.core._types import Model, DataSource
from pnpxai.detector import ModelArchitecture
from pnpxai.detector._core import NodeInfo

from .rules import AttentionHeadRule, LayerNormRule
from .utils import list_args_for_stack


class CaptumLikeZennit:
    def __init__(self, model: Model):
        self.model = model
    
    @staticmethod    
    def _replace_add_func_with_mod(model):
        # get model architecture to manipulate
        ma = ModelArchitecture(model)

        # find add functions from model and replace them to modules
        add_func_nodes = ma.find_node(
            lambda n: n.operator is add and all(isinstance(arg, NodeInfo) for arg in n.args),
            all = True
        )
        if not add_func_nodes:
            return model
        
        warnings.warn(
            f"\n[LRP] Warning: {len(add_func_nodes)} add operations in function detected. Automatically changed to modules."
        )
        
        # replace
        for add_func_node in add_func_nodes:
            add_mod_node = ma.replace_node(
                add_func_node,
                NodeInfo.from_module(Sum()),
            )
            stack_node = ma.insert_node(
                NodeInfo.from_function(torch.stack, dim=-1),
                add_mod_node,
                before = True,
            )
            _ = ma.insert_node(
                NodeInfo.from_function(list_args_for_stack),
                stack_node,
                before = True,
            )
        return ma.traced_model
    
    @staticmethod
    def _has_attn_layer(model):
        ma = ModelArchitecture(model)
        attn_node = ma.find_node(lambda n: isinstance(n.operator, nn.MultiheadAttention))
        return attn_node is not None
                
    def attribute(
        self,
        inputs: DataSource,
        target: TargetType,
        attributor_type: Any = Gradient,
        canonizers: Any = [SequentialMergeBatchNorm()],
        composite_type: Any = LayerMapComposite,
        additional_composite_args: Dict = {},
        n_classes: int = 1000,
    ) -> List[torch.Tensor]:
        model = self._replace_add_func_with_mod(self.model)
        if isinstance(composite_type, type(LayerMapComposite)):
            additional_composite_args["layer_map"] = layer_map_base()
        composite = composite_type(canonizers=canonizers, **additional_composite_args)
        if self._has_attn_layer(self.model):
            # [GH] implementation trick
            # nn.MultiheadAttention is an instance of zennit.type.Activation
            # to which Pass rule is assigned by layer_map_base(). By simply
            # change the order of composite.layer_map, assign AttentionHeadRule
            composite.layer_map = [
                (nn.LayerNorm, LayerNormRule()),
                (nn.MultiheadAttention, AttentionHeadRule()),
            ] + composite.layer_map

        if isinstance(target, int):
            targets = [target]
        elif torch.is_tensor(target):
            targets = target.tolist()
        else:
            raise Exception(f"[LRP] Unsupported target type: {type(target)}")

        with attributor_type(model=model, composite=composite) as attributor:
            _, relevance = attributor(inputs, torch.eye(n_classes)[targets])
        return relevance
