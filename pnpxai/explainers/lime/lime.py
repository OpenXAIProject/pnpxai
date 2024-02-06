from typing import Any, Union, Tuple, Optional, Dict

from captum.attr import Lime as LimeCaptum
from captum._utils.typing import BaselineType, TargetType

from torch import Tensor

from pnpxai.explainers.utils.feature_mask import get_default_feature_mask
from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer


class Lime(Explainer):
    """
    Computes LIME explanations for a given model.

    Attributes:
    - model (Model): The model for which LIME explanations are computed.
    - source (LimeCaptum): The LIME source for explanations.
    """
    def __init__(self, model: Model):
        """
        Initializes a Lime object.

        Args:
        - model (Model): The model for which LIME explanations are computed.
        """
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
        """
        Computes LIME attributions for the given inputs.

        Args:
        - inputs (DataSource): The input data.
        - targets (TargetType): The target labels for the inputs (default: None).
        - baselines (BaselineType): The baselines for attribution (default: None).
        - additional_forward_args (Any): Additional arguments for forward pass (default: None).
        - feature_mask (Union[None, Tensor, Tuple[Tensor, ...]]): The feature mask (default: None).
        - n_samples (int): Number of samples (default: 25).
        - perturbations_per_eval (int): Number of perturbations per evaluation (default: 1).
        - return_input_shape (bool): Whether to return input shape (default: True).
        - show_progress (bool): Whether to show progress (default: False).

        Returns:
        - LIME attributions.
        """        
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
