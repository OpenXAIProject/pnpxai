from typing import Optional
from captum._utils.typing import TargetType

from pnpxai.core._types import Model, DataSource, Task, Tensor
from pnpxai.explainers._explainer import Explainer
from .lrp_zennit import LRPZennit


class LRP(Explainer):
    """
    Computes Layer-wise Relevance Propagation (LRP) explanations for a given model.

    Args:
        model (Model): The model for which LRP explanations are computed.

    Attributes:
        source (LRPZennit): The LRP source for explanations.
    """

    def __init__(self, model: Model):
        super(LRP, self).__init__(model)
        self.source = LRPZennit(model)

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        epsilon: float = .25,
        n_classes: Optional[int] = None,
    ) -> DataSource:
        """
        Computes LRP attributions for the given inputs.

        Args:
            inputs (DataSource): The input data (N x C x H x W).
            targets (TargetType): The target labels for the inputs (N x 1, default: None).
            epsilon (float): The epsilon value for numerical stability (default: 0.25).
            n_classes (Optional[int]): Number of classes (default: None).
        """        
        attributions = self.source.attribute(
            inputs=inputs,
            targets=targets,
            epsilon=epsilon,
            n_classes=n_classes,
        )
        return attributions

    def format_outputs_for_visualization(
        self,
        inputs: Tensor,
        targets: Tensor,
        explanations: Tensor,
        task: Task,
        kwargs: Optional[dict] = None,
    ):
        explanations = explanations.transpose(-1, -3) \
            .transpose(-2, -3)
        return super().format_outputs_for_visualization(
            inputs=inputs,
            targets=targets,
            explanations=explanations,
            task=task,
            kwargs=kwargs
        )
