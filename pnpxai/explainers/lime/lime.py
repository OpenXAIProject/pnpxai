from typing import Any, Union, Tuple

from captum.attr import Lime as LimeCaptum
from captum._utils.typing import BaselineType, TargetType

from torch import Tensor

from pnpxai.explainers.utils.feature_mask import get_default_feature_mask
from pnpxai.core._types import Model, DataSource
from pnpxai.explainers._explainer import Explainer


class Lime(Explainer):
    def __init__(self, model: Model):
        super().__init__(model=model)
        self.source = LimeCaptum(model)

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
    ):
        if feature_mask is None:
            feature_mask = get_default_feature_mask(inputs, self.device)

        return self.source.attribute(
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
