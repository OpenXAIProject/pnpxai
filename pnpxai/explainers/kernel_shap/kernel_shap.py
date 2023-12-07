from typing import Any, List, Tuple

from torch import Tensor

from captum.attr import KernelShap as KernelShapeCaptum
from captum._utils.typing import BaselineType, TargetType

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer
from pnpxai.explainers.utils.feature_mask import get_default_feature_mask

from typing import Union, Optional, Dict


class KernelShap(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.source = KernelShapeCaptum(model)

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False
    ) -> List[Tensor]:
        if feature_mask is None:
            feature_mask = get_default_feature_mask(inputs, self.device)

        attributions = self.source.attribute(
            inputs=inputs,
            baselines=baselines,
            target=targets,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
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
        explanations = explanations.transpose(-1, -3)\
            .transpose(-2, -3)
        return super().format_outputs_for_visualization(
            inputs=inputs,
            targets=targets,
            explanations=explanations,
            task=task,
            kwargs=kwargs
        )
