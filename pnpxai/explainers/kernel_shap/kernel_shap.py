from typing import Any, List, Tuple

from torch import Tensor

import numpy as np
from captum.attr import KernelShap as KernelShapCaptum
from captum._utils.typing import BaselineType, TargetType
from shap import KernelExplainer
from shap.utils._legacy import DenseData, kmeans
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator

from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer
from pnpxai.explainers.utils.feature_mask import get_default_feature_mask

from typing import Union, Optional, Dict, Callable


class KernelShap(Explainer):
    """
    Computes KernelSHAP explanations for a given model.

    Args:
        model (Model): The model for which KernelSHAP explanations are computed.

    Attributes:
        source (KernelShapCaptum): The KernelSHAP source for explanations.
    """

    def __init__(self, model: Model):
        super().__init__(model)
        self.source = KernelShapCaptum(model)

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
        """
        Computes KernelSHAP attributions for the given inputs.

        Args:
            inputs (DataSource): The input data (N x C x H x W).
            targets (TargetType): The target labels for the inputs (N x 1, default: None).
            baselines (BaselineType): The baselines for attribution (default: None).
            additional_forward_args (Any): Additional arguments for forward pass (default: None).
            feature_mask (Union[None, Tensor, Tuple[Tensor, ...]]): The feature mask (default: None).
            n_samples (int): Number of samples (default: 25).
            perturbations_per_eval (int): Number of perturbations per evaluation (default: 1).
            return_input_shape (bool): Whether to return input shape (default: True).
            show_progress (bool): Whether to show progress (default: False).
        """
        if feature_mask is None:
            feature_mask = get_default_feature_mask(inputs, self.device)
        assert len(feature_mask.unique()) > 1, "The number of feature_mask must be more than one"
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


class TabKernelShap(Explainer):

    kmeans = kmeans
    
    def __init__(self, model: Callable, bg_data: DenseData, mode: str = "classification"):
        super().__init__(model)
        if isinstance(model, XGBClassifier) or isinstance(model, BaseEstimator):
            def wrapper(model):
                def callable(inputs):
                    return model.predict_proba(inputs)
                return callable
            
        elif isinstance(model, XGBRegressor) or isinstance(model, BaseEstimator):
            def wrapper(model):
                def callable(inputs):
                    return model.predict(inputs)
                return callable
        else:
            raise ValueError(f"The model is unsupported. The type of the model is {type(model)}")
        
        model = wrapper(model)

        if not isinstance(model, Callable):
            raise ValueError("The model must be callable")

        self.source = KernelExplainer(model, bg_data)
        self.mode = mode

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        n_samples: int = 25,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        
        if self.mode == "classification":
            if len(inputs) != len(targets):
                raise ValueError("The number of inputs and targets must be the same")
        
        attributions = self.source.shap_values(
            X=inputs,
            nsamples=n_samples,
            show=show_progress
        )
        if self.mode == "classification":
            attributions = attributions[np.arange(attributions.shape[0]),:,targets]
        
        return attributions

    def format_outputs_for_visualization(
        self,
        inputs: DataSource,
        targets: DataSource,
        explanations: DataSource,
        task: Task,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        return None
