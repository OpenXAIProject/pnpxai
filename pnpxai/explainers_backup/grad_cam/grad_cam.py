from typing import List, Dict, Optional, Any, Union
from torch import nn, Tensor

from captum.attr import LayerAttribution
from captum.attr import LayerGradCam as GradCamCaptum
from captum._utils.typing import TargetType
from zennit.types import SubclassMeta

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.detector_backup import ModelArchitecture
from pnpxai.explainers_backup._explainer import Explainer

class Pool2d(metaclass=SubclassMeta):
    __subclass__ = (
        nn.AvgPool2d,
        nn.AdaptiveAvgPool2d,
        nn.MaxPool2d,
        nn.AdaptiveMaxPool2d,
    )

class GradCam(Explainer):
    """
    Computes Grad-CAM explanations for a given model.

    Args:
        model (Model): The model for which Grad-CAM explanations are computed.

    Attributes:
        source (GradCamCaptum): The Grad-CAM source for explanations.
    """
    def __init__(self, model: Model):
        super().__init__(model=model)
        self.source = GradCamCaptum(
            self.model,
            layer=self._find_target_layer(),
        )
    
    def _find_target_layer(self) -> Optional[Union[nn.Conv2d, Pool2d]]:
        ma = ModelArchitecture(self.model)
        conv_nodes = ma.find_node(lambda n: isinstance(n.operator, nn.Conv2d), get_all=True)
        assert conv_nodes, "Must have nn.Conv2d"
        last_conv_node = conv_nodes[-1]
        pool_nodes = ma.find_node(
            lambda n: isinstance(n.operator, Pool2d),
            root=last_conv_node,
            get_all=True,
        )
        if pool_nodes:
            return getattr(self.model, pool_nodes[-1].prev.owning_module)
        return last_conv_node.operator

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
        # attr_dim_summation: bool = True,
    ) -> List[Tensor]:
        """
        Computes Grad-CAM attributions for the given inputs.

        Args:
            inputs (DataSource): The input data  (N x C x H x W).
            targets (TargetType): The target labels for the inputs (N x 1, default: None).
            additional_forward_args (Any): Additional arguments for forward pass (default: None).
            attribute_to_layer_input (bool): Whether to attribute to layer input (default: False).
            relu_attributions (bool): Whether to compute ReLU attributions (default: False).
        """
        attributions = self.source.attribute(
            inputs=inputs,
            target=targets,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            relu_attributions=relu_attributions,
            # attr_dim_summation=attr_dim_summation,
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
