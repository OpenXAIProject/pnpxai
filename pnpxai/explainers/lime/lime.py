from typing import Any, Union, Tuple, Optional, Dict, List, Callable

from captum.attr import Lime as LimeCaptum
from captum._utils.typing import BaselineType, TargetType

from torch import Tensor
import numpy as np

from pnpxai.explainers.utils.feature_mask import get_default_feature_mask
from pnpxai.core._types import Model, DataSource, Task
from pnpxai.explainers._explainer import Explainer

from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import ClassifierMixin, RegressorMixin
from xgboost import XGBClassifier, XGBRegressor


class Lime(Explainer):
    """
    Computes LIME explanations for a given model.

    Args:
        model (Model): The model for which LIME explanations are computed.

    Attributes:
        source (LimeCaptum): The LIME source for explanations.
    """
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
        """
        Computes LIME attributions for the given inputs.

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



class TabLime(Explainer):
    def __init__(self, model: Model, bg_data: np.ndarray, categorical_features: List[int] = None, mode: str = 'classification'):
        super().__init__(model=model)
        self.source = None
        self.bg_data = bg_data
        self.categorical_features = categorical_features
        self.mode = mode
        if isinstance(model, ClassifierMixin) or isinstance(model, XGBClassifier):
            self.predict_fn = model.predict_proba
        elif isinstance(model, RegressorMixin) or isinstance(model, XGBRegressor):
            self.predict_fn = model.predict
        elif isinstance(model, Callable):
            self.predict_fn = model
        else:
            raise ValueError("The model must be callable")

    def attribute(
        self,
        inputs: DataSource,
        targets: TargetType = None,
        n_samples: int = 25,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        
        self.source = LimeTabularExplainer(
            self.bg_data,
            categorical_features=self.categorical_features, 
            verbose=False, 
            mode=self.mode
        )
        
        attributions = []
        for _input in inputs:
            res = self.source.explain_instance(
                data_row=_input, 
                predict_fn=self.predict_fn,
                num_samples=n_samples,
                num_features=len(_input),
            )
            sorted_data = sorted(res.as_map()[1], key=lambda x: x[0])
            values = np.array([x[1] for x in sorted_data])
            attributions.append(values)

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
