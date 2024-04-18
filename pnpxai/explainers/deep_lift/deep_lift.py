from typing import Any, Union, Tuple, Optional, Dict, Callable

from captum.attr import DeepLift as DeepLiftCaptum
from captum._utils.typing import BaselineType, TargetType

from torch import Tensor

from pnpxai.explainers.utils.feature_mask import get_default_feature_mask
from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer


class DeepLift(Explainer):
    """
    Computes DeepLift explanations for a given model.

    Args:
        model (Model): The model for which DeepLift explanations are computed.

    Attributes:
        source (DeepLiftCaptum): The DeepLift source for explanations.
    """

    def __init__(self, model: Model):
        super().__init__(model=model)
        self.source = DeepLiftCaptum(model)

    def attribute(
        self,
        inputs: DataSource,
        baselines: BaselineType = None,
        targets: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: Union[None,
                                       Callable[..., Tuple[Tensor, ...]]] = None,
    ):
        """
        Computes DeepLift attributions for the given inputs.

        Args:
            inputs (DataSource): The input data
            targets (TargetType): The target labels for the inputs
            baselines (BaselineType): The baselines for attribution (default: None).
            additional_forward_args (Any): Additional arguments for forward pass (default: None).
            return_convergence_delta (bool): Whether to return convergence delta (default: False).
            custom_attribution_func (Union[None, Callable[..., Tuple[Tensor, ...]]]): Custom attribution function (default: None).
        """

        return self.source.attribute(
            inputs=inputs,
            baselines=baselines,
            target=targets,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
            custom_attribution_func=custom_attribution_func
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
