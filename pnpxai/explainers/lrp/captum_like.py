import warnings
from _operator import add
from typing import Any, List, Dict

import torch
from captum._utils.typing import TargetType

from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlus
from zennit.layer import Sum

from pnpxai.core._types import Model, DataSource
from pnpxai.detector import ModelArchitecture
from pnpxai.detector._core import NodeInfo

from .utils import list_args_for_stack

class CaptumLikeZennit:
    def __init__(self, model: Model):
        self.model = model
    
    def _replace_add_func_with_mod(self):
        # get model architecture to manipulate
        ma = ModelArchitecture(self.model)

        # find add functions from model and replace them to modules
        add_func_nodes = ma.find_node(
            lambda n: n.operator is add and all(isinstance(arg, NodeInfo) for arg in n.args),
            all = True
        )
        if add_func_nodes:
            return
        
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
        self.model = ma.traced_model
            
    def attribute(
        self,
        inputs: DataSource,
        target: TargetType,
        attributor_type: Any = Gradient,
        canonizers: Any = [SequentialMergeBatchNorm()],
        composite_type: Any = EpsilonPlus,
        additional_composite_args: Dict = {},
        n_classes: int = 1000,
    ) -> List[torch.Tensor]:
        self._replace_add_func_with_mod()
        composite = composite_type(canonizers=canonizers, **additional_composite_args)

        if isinstance(target, int):
            targets = [target]
        elif torch.is_tensor(target):
            targets = target.tolist()
        else:
            raise Exception(f"[LRP] Unsupported target type: {type(target)}")

        with attributor_type(model=self.model, composite=composite) as attributor:
            _, relevance = attributor(inputs, torch.eye(n_classes)[targets])
        return relevance