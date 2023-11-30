import warnings
from _operator import add
from typing import Any, List, Dict, Optional

import torch
from torch import nn
from captum._utils.typing import TargetType

from zennit.core import Composite
from zennit.attribution import Attributor, Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import LayerMapComposite, layer_map_base
from zennit.rules import Epsilon
from zennit.layer import Sum
from zennit.types import Linear

from pnpxai.core._types import Model, DataSource
from pnpxai.detector import ModelArchitecture
from pnpxai.detector._core import NodeInfo

from .rules import AttentionHeadRule, LayerNormRule
from .utils import list_args_for_stack

_layer_map_base = [
    (Linear, Epsilon()),
    (nn.LayerNorm, LayerNormRule()),
    (nn.MultiheadAttention, AttentionHeadRule(1e-9)),
] + layer_map_base()

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
    
    # @staticmethod
    # def _get_attn_layers(model):
    #     ma = ModelArchitecture(model)
    #     attn_nodes = ma.find_node(lambda n: isinstance(n.operator, nn.MultiheadAttention), all=True)
    #     return [(n.operator, n.kwargs) for n in attn_nodes]
    
    def attribute(
        self,
        inputs: DataSource,
        target: TargetType,
        attributor_type: Optional[type[Attributor]] = None,
        composite: Optional[Composite] = None,
        n_classes: int = 1000,
    ) -> List[torch.Tensor]:
        model = self._replace_add_func_with_mod(self.model)
        if isinstance(target, int):
            targets = [target]
        elif torch.is_tensor(target):
            targets = target.tolist()
        else:
            raise Exception(f"[LRP] Unsupported target type: {type(target)}")
        
        if attributor_type is None:
            attributor_type = Gradient
        if composite is None:
            canonizers = [SequentialMergeBatchNorm()]
            composite = LayerMapComposite(layer_map=_layer_map_base, canonizers=canonizers)

        # import pdb; pdb.set_trace()
        with attributor_type(model=model, composite=composite) as attributor:            
            _, relevance = attributor(inputs, torch.eye(n_classes)[targets])
        return relevance
