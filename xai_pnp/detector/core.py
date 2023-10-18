from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# [TODO] split LayerInfo to [NodeBase, NodeModule, NodeFunction, NodeAttr]
@dataclass
class LayerInfo:
    name: str
    module_name: str
    class_name: str
    output_size: Optional[Tuple]
    grad_fn: Optional[str]


# [TODO] split Graph from ModelArchitecture
# class Graph:
#     def __init__(self, nodes: List=[])
#         self.node = nodes
#     def crud(self):
#         pass

class ModelArchitecture:
    def __init__(self, data: Optional[List[LayerInfo]]=None):
        self.data = data if data else [] # naming as "graph" will be better

    # CRUD control on nodes: [TODO] move this part to Graph
    # create
    def _add_data(self, data: Dict): # "_add_node"
        self.data.append(LayerInfo(**data))

    # Detection logics
    # global logics
    # [global logic1] Is this model containing at least a supported module to target?
    def _contains_supported_module(self, supported_module, exclude=[]):
        # [TODO] test subtree in tree, not module name
        return any(
            info.module_name in supported_module
            for i, info in enumerate(self.data)
            if i not in exclude
        )

    # [TODO] additional global logics to detect target: function level detection logics
    def _additional_logics(self):
        return True

    # Detections
    def is_convolution(self):
        if not self._contains_supported_module(SUPPORTED_CONV_MODULES, exclude=[0]):
            return False
        valid = self._additional_logics()
        # [TODO] additional conv-specific logics
        return valid

    def is_transformer(self):
        # exclude the first projection layer
        if not self._contains_supported_module(SUPPORTED_TRANSFORMER_MODULES):
            return False
        valid = self._additional_logics()
        return valid


SUPPORTED_CONV_MODULES = {
    "torch.nn.modules.conv",
}

SUPPORTED_TRANSFORMER_MODULES = {
    "torch.nn.modules.transformer",
    "torchvision.models.vision_transformer",
    "transformers.models.vit.modeling_vit",
    "timm.models.vision_transformer",
}
