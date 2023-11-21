from typing import Dict, List, Sequence, Optional
from torch import nn

from captum.attr import GuidedGradCam as GuidedGradCamCaptum

from pnpxai.core._types import Model
from pnpxai.explainers._explainer import Explainer

class GuidedGradCam(Explainer):
    def __init__(self, model: Model):
        super().__init__(
            source = GuidedGradCamCaptum,
            model = model,
        )

    @property
    def _attributor_arg_keys(self) -> List[str]:
        return ["layer"]
    
    def get_default_additional_kwargs(self) -> Dict:
        return {
            "layer": self._find_last_conv_layer(self.model.modules()),
            "additional_forward_args": None,
            "interpolate_mode": "nearest",
            "attribute_to_layer_input": False,
        }
    
    # TODO: integrate with ..utils.operation_graph
    def _find_last_conv_layer(self, modules: Sequence[nn.Module]) -> Optional[nn.Conv2d]:
        last_conv = None
        for module in modules:
            if isinstance(module, nn.Conv2d):
                last_conv = module

            submodules = list(module.children())
            if len(submodules) > 0:
                last_conv = self._find_last_conv_layer(submodules)
        
        return last_conv