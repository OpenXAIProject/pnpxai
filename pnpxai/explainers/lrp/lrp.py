from typing import Any, Optional, Callable
from captum._utils.typing import TargetType

from pnpxai.core._types import Model, DataSource, Task, Tensor
from pnpxai.explainers._explainer import Explainer
from .lrp_zennit import LRPZennit, Attributor, Composite


class LRP(Explainer):
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
