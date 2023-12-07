from typing import Any, Optional, Callable
from captum._utils.typing import TargetType

from pnpxai.core._types import Model, DataSource, Task
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
        epsilon: float = 1e-6,
        n_classes: Optional[int] = 1000,
    ) -> DataSource:
        if n_classes is None:
            n_classes = self.model(inputs).shape[-1]
        attributions = self.source.attribute(
            inputs=inputs,
            targets=targets,
            epsilon=epsilon,
            n_classes=n_classes,
        )

        return attributions

    def format_outputs_for_visualization(
        self,
        data: DataSource,
        explanations: DataSource,
        input_extractor: Callable[[Any], Any],
        target_extractor: Callable[[Any], Any],
        task: Task,
        kwargs,
    ):
        explanations = explanations.permute((1, 2, 0))
        return super().format_outputs_for_visualization(
            data=data,
            input_extractor=input_extractor,
            target_extractor=target_extractor,
            explanations=explanations,
            task=task,
            kwargs=kwargs
        )
