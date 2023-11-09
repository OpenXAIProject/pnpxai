from typing import List, Dict

from torch import nn, Tensor
from open_xai.explainers.rap import rules

SUPPORTED_LAYERS: Dict[type[nn.Module], type[rules.RelProp]] = {
    nn.Sequential: rules.Sequential,
    nn.ReLU: rules.ReLU,
    nn.Dropout: rules.Dropout,
    nn.MaxPool2d: rules.MaxPool2d,
    nn.AdaptiveAvgPool2d: rules.AdaptiveAvgPool2d,
    nn.AvgPool2d: rules.AvgPool2d,
    nn.BatchNorm2d: rules.BatchNorm2d,
    nn.Linear: rules.Linear,
    nn.Conv2d: rules.Conv2d,
}


class RelativeAttributePropagation():
    def __init__(self, model: nn.Module):
        self.model = model
        self._original_model_state_dict = model.state_dict()
        self.layers = self._get_layers(self.model)
        
        self._assign_rules()
        self._init_rule_hooks()

    def _get_layers(self, model: nn.Module) -> List[nn.Module]:
        layers = []
        for layer in model.children():
            layers.append(layer)
            if len(list(layer.children())) > 0:
                layers += self._get_layers(layer)

        return layers

    def _assign_rules(self):
        for layer in self.layers:
            if type(layer) in SUPPORTED_LAYERS and not (hasattr(layer, 'rule')):
                layer.rule = SUPPORTED_LAYERS[type(layer)](layer)

    def _init_rule_hooks(self):
        for layer in self.layers:
            if hasattr(layer, 'rule'):
                layer.register_forward_hook(layer.rule.forward_hook)

    def relprop(self, r: Tensor) -> Tensor:
        for layer in self.layers[::-1]:
            r = layer.rule.relprop(r)

        return r
