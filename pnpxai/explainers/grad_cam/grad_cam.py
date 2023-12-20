from typing import List, Dict, Optional, Any, Union
from torch import nn, Tensor

from captum.attr import LayerAttribution
from captum.attr import LayerGradCam as GradCamCaptum
from captum._utils.typing import TargetType
from zennit.types import SubclassMeta

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.detector import ModelArchitecture
from pnpxai.explainers._explainer import Explainer

class Pool2d(metaclass=SubclassMeta):
    __subclass__ = (
        nn.AvgPool2d,
        nn.AdaptiveAvgPool2d,
        nn.MaxPool2d,
        nn.AdaptiveMaxPool2d,
    )

class GradCam(Explainer):
    def __init__(self, model: Model):
        super().__init__(model=model)
        self.source = GradCamCaptum(
            self.model,
            layer=self._find_target_layer(),
        )
    
    def _find_target_layer(self) -> Optional[Union[nn.Conv2d, Pool2d]]:
        ma = ModelArchitecture(self.model)
        conv_nodes = ma.find_node(lambda n: isinstance(n.operator, nn.Conv2d), all=True)
        assert conv_nodes, "Must have nn.Conv2d"
        last_conv_node = conv_nodes[-1]
        pool_nodes = ma.find_node(
            lambda n: isinstance(n.operator, Pool2d),
            root=last_conv_node,
            all=True,
        )
        assert pool_nodes, "Must have pooling layer"
        target_module = getattr(self.model, pool_nodes[-1].prev.owning_module)
        return target_module

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
        attr_dim_summation: bool = True,
    ) -> List[Tensor]:
        attributions = self.source.attribute(
            inputs=inputs,
            target=targets,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            relu_attributions=relu_attributions,
            attr_dim_summation=attr_dim_summation,
        )
        upsampled = LayerAttribution.interpolate(attributions, inputs.shape[2:], "bilinear")
        return upsampled

    def format_outputs_for_visualization(
        self,
        inputs: DataSource,
        targets: DataSource,
        explanations: DataSource,
        task: Task,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        explanations = explanations.transpose(-1, -3)\
            .transpose(-2, -3)
        return super().format_outputs_for_visualization(
            inputs=inputs,
            targets=targets,
            explanations=explanations,
            task=task,
            kwargs=kwargs
        )