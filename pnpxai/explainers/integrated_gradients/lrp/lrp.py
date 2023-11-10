from collections import defaultdict
from typing import Dict

from torch import nn
from captum.attr import LRP as CaptumLRP
from captum.attr._utils.lrp_rules import PropagationRule
from captum.attr._utils.custom_modules import Addition_Module

from open_xai.detector import ModelArchitecture
from open_xai.detector._core import NodeInfo


SUPPORTED_NON_LINEAR_LAYERS = [nn.ReLU, nn.Dropout, nn.Tanh]

class _LRPBase(CaptumLRP):
    def __init__(
            self,
            model: nn.Module,
            layer_map: Dict[nn.Module, PropagationRule]
        ):
        self.model_architecture = self._trace_and_modify_model(model)
        super().__init__(model=self.model_architecture.traced_model)
        self.layer_map = layer_map
    
    def _trace_and_modify_model(self, model) -> ModelArchitecture:
        # check addition functions and replace them to module
        from _operator import add
        
        # trace
        ma = ModelArchitecture(model)

        # modify
        add_nodes = ma.find_node(lambda n: n.operator is add, all=True)
        for n in add_nodes:
            new_node = NodeInfo.from_module(Addition_Module())
            ma.replace_node(n, new_node)
        return ma

    # override
    def _get_layers(self, model) -> None:        
        self.layers = [
            n.operator for n in self.model_architecture.list_nodes()
            if n.opcode == "call_module"
        ]

    # override
    def _check_and_attach_rules(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "rule"):
                if layer.rule is None:
                    continue
                layer.activations = {}
                layer.rule.relevance_input = defaultdict(list)
                layer.rule.relevance_output = {}
            elif type(layer) in self.layer_map.keys():
                layer.activations = {}
                layer.rule = self.layer_map[type(layer)]()
                layer.rule.relevance_input = defaultdict(list)
                layer.rule.relevance_output = {}
            elif type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                layer.rule = None
            else:
                raise TypeError(
                    (
                        f"Module of type {type(layer)} has no rule defined and no"
                        "default rule exists for this module type. Please, set a rule"
                        "explicitly for this module and assure that it is appropriate"
                        "for this type of layer."
                    )
                )


