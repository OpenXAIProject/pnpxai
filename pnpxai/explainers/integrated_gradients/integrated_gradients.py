from typing import Sequence, Any, Union, Literal, Optional, Dict
from torch import Tensor

from captum.attr import IntegratedGradients as IntegratedGradientsCaptum
from captum._utils.typing import BaselineType, TargetType

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer


class IntegratedGradients(Explainer):
    def __init__(self, model: Model):
        super().__init__(model=model)
        self.source = IntegratedGradientsCaptum(self.model)

    def attribute(
        self,
        inputs: DataSource,
        target: TargetType = None,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: Literal[True] = False
    ) -> Tensor:
        attributions = self.source.attribute(
            inputs,
            baselines,
            target,
            additional_forward_args,
            n_steps,
            method,
            internal_batch_size,
            return_convergence_delta
        )

        return attributions

    def format_outputs_for_visualization(
        self,
        inputs: DataSource,
        targets: DataSource,
        explanations: DataSource,
        task: Task,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if kwargs.get('return_convergence_delta', False):
            explanations = explanations[0]
        explanations = explanations.permute((1, 2, 0))
        return super().format_outputs_for_visualization(
            inputs=inputs,
            targets=targets,
            explanations=explanations,
            task=task,
            kwargs=kwargs
        )
