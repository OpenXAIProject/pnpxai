from typing import Any, List, Sequence, Optional

from torch import Tensor, nn

from captum.attr import LayerGradCam
from captum._utils.typing import TargetType
from plotly import express as px
from plotly.graph_objects import Figure

from open_xai.core._types import Model, DataSource
from open_xai.explainers._explainer import Explainer
from open_xai.detector import ModelArchitecture

class GradCam(Explainer):
    def __init__(self, model: Model, layer: Optional[Model]=None):
        super().__init__(model)
        self.layer = layer if layer else self._locate_candidate_layer(model)
        self.method = LayerGradCam(forward_func=self.model, layer=self.layer)
    
    def _locate_candidate_layer(self, model):
        ma = ModelArchitecture(model)

        # check whether conv exists
        conv_filter = lambda n: isinstance(n.operator, nn.Conv2d)
        first_conv_node = ma.find_node(conv_filter)
        assert first_conv_node, "GradCam applicable layer not found."

        # check whether the conv is pooled
        pool_filter = lambda n: (
            n.opcode == "call_module"
            and n.operator.__module__ == "torch.nn.modules.pooling"
        )
        pool_nodes = ma.find_node(pool_filter, root=first_conv_node, all=True)
        assert pool_nodes, "GradCam applicable layer not found"

        # get the final pooling node
        final_pool_node = pool_nodes[-1]

        # cam target node is before the final pooling node
        cam_target_node = final_pool_node.prev

        # get the layer
        accessible_name = next(reversed(cam_target_node.meta["nn_module_stack"]))
        m = model
        for s in accessible_name.split("."):
            m = getattr(m, s)
        return m

    def run(
        self,
        data: DataSource,
        target: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
    ) -> List[Tensor]:
        attributions = []

        if type(data) is Tensor:
            data = [data]

        for datum in data:
            attributions.append(self.method.attribute(
                datum,
                target,
                additional_forward_args,
                attribute_to_layer_input = attribute_to_layer_input,
                relu_attributions = relu_attributions,
            ))

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]
