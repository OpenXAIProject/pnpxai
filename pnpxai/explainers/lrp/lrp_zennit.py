import warnings
from _operator import add
from typing import List, Optional

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
from pnpxai.explainers._explainer import Explainer
from .rules import AttentionHeadRule, LayerNormRule
from .utils import list_args_for_stack

DEFAULT_LAYER_MAP = [
    (Linear, Epsilon()),
    (nn.LayerNorm, LayerNormRule()),
    (nn.MultiheadAttention, AttentionHeadRule(1e-9)),
] + layer_map_base()

class LRPZennit(Explainer):
    def __init__(self, model: Model):
        super(LRPZennit, self).__init__(model)

    def _replace_add_func_with_mod(self):
        # get model architecture to manipulate
        ma = ModelArchitecture(self.model)

        # find add functions from model and replace them to modules
        add_func_nodes = ma.find_node(
            lambda n: n.operator is add and all(
                isinstance(arg, NodeInfo) for arg in n.args),
            all=True
        )
        if add_func_nodes:
            self.__set_model_with_functional_nodes(ma, add_func_nodes)

    def __set_model_with_functional_nodes(self, ma: ModelArchitecture, functional_nodes: List[NodeInfo]):
        warnings.warn(
            f"\n[LRP] Warning: {len(functional_nodes)} add operations in function detected. Automatically changed to modules."
        )

        # replace
        for add_func_node in functional_nodes:
            add_mod_node = ma.replace_node(
                add_func_node,
                NodeInfo.from_module(Sum()),
            )
            stack_node = ma.insert_node(
                NodeInfo.from_function(torch.stack, dim=-1),
                add_mod_node,
                before=True,
            )
            _ = ma.insert_node(
                NodeInfo.from_function(list_args_for_stack),
                stack_node,
                before=True,
            )
        self.model = ma.traced_model

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType,
        attributor_type: Optional[type[Attributor]] = None,
        composite: Optional[Composite] = None,
        n_classes: int = 1000,
    ) -> List[torch.Tensor]:
        model = self._replace_add_func_with_mod(self.model)
        if isinstance(targets, int):
            targets = [targets]
        elif torch.is_tensor(targets):
            targets = targets.tolist()
        else:
            raise Exception(f"[LRP] Unsupported target type: {type(targets)}")
        
        if attributor_type is None:
            attributor_type = Gradient
        if composite is None:
            canonizers = [SequentialMergeBatchNorm()]
            composite = LayerMapComposite(layer_map=DEFAULT_LAYER_MAP, canonizers=canonizers)

        with attributor_type(model=model, composite=composite) as attributor:            
            _, relevance = attributor(inputs, torch.eye(n_classes)[targets])
        return relevance
