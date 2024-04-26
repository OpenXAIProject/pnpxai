from _operator import add
from importlib import util
from typing import List

import torch
from torch import nn
from captum._utils.typing import TargetType

from zennit.attribution import Gradient
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
from .utils import list_args_for_stack, LRPTracer

class LRPZennit(Explainer):
    def __init__(self, model: Model):
        super(LRPZennit, self).__init__(model)

    def _replace_add_func_with_mod(self):
        # get model architecture to manipulate
        ma = ModelArchitecture(model=self.model, tracer=LRPTracer())

        # find add functions from model and replace them to modules
        add_func_nodes = ma.find_node(
            lambda n: n.operator is add and all(
                isinstance(arg, NodeInfo) for arg in n.args),
            get_all=True
        )

        if add_func_nodes:
            traced_model = self.__get_model_with_functional_nodes(ma, add_func_nodes)
            return traced_model
        return self.model

    def __get_model_with_functional_nodes(self, ma: ModelArchitecture, functional_nodes: List[NodeInfo]):
        # warnings.warn(
        #     f"\n[LRP] Warning: {len(functional_nodes)} add operations in function detected. Automatically changed to modules."
        # )

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
        return ma._traced_model

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType,
        epsilon: float = 1e-6,
        n_classes: int = None,
    ) -> List[torch.Tensor]:
        model = self._replace_add_func_with_mod()
        # if isinstance(targets, int):
        #     targets = [targets] * len(inputs)
        # elif torch.is_tensor(targets):
        #     targets = targets.tolist()
        # else:
        #     raise Exception(f"[LRP] Unsupported target type: {type(targets)}")
        # if n_classes is None:
        #     n_classes = self.model(inputs).shape[-1]
        
        additional_layer_map = [
            (Linear, Epsilon(epsilon=epsilon)),
            (nn.MultiheadAttention, AttentionHeadRule(stabilizer=epsilon)),
            (nn.LayerNorm, LayerNormRule(stabilizer=epsilon)),
        ]
        canonizers = [SequentialMergeBatchNorm()]
        if util.find_spec("timm"):
            from .timm import (
                VisionTransformerAttention,
                VisionTransformerAttentionCanonizer,
            )
            additional_layer_map.append((VisionTransformerAttention, AttentionHeadRule(stabilizer=epsilon)))
            canonizers.append(VisionTransformerAttentionCanonizer())
        layer_map = additional_layer_map + layer_map_base()
        composite = LayerMapComposite(layer_map=layer_map, canonizers=canonizers)

        with Gradient(model=model, composite=composite) as attributor:
            # _, relevance = attributor(inputs, torch.eye(n_classes)[targets].to(self.device))
            _, relevance = attributor(inputs, targets.to(self.device))
        return relevance
