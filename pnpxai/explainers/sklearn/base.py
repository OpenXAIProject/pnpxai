from abc import abstractmethod
from typing import Union, Callable

from sklearn.base import ClassifierMixin, RegressorMixin
from xgboost import XGBClassifier, XGBRegressor

from pnpxai.core._types import SklearnModel, XGBModel
from pnpxai.explainers.base import ABC
import pandas as pd


class SklearnExplainer(ABC):
    EXPLANATION_TYPE = 'attribution'

    def __init__(
        self,
        model: Union[SklearnModel, XGBModel],
        background_data: pd.DataFrame,
        **kwargs,
    ):
        self.model = model
        self.background_data = background_data

    @property
    def _predict_fn(self):
        if isinstance(self.model, ClassifierMixin) or isinstance(self.model, XGBClassifier):
            return self.model.predict_proba
        elif isinstance(self.model, RegressorMixin) or isinstance(self.model, XGBRegressor):
            return self.model.predict
        elif isinstance(self.model, Callable):
            return self.model
        else:
            raise ValueError("The model must be callable")

    @property
    def mode(self):
        if isinstance(self.model, ClassifierMixin) or isinstance(self.model, XGBClassifier):
            return 'classification'
        elif isinstance(self.model, RegressorMixin) or isinstance(self.model, XGBRegressor):
            return 'regression'

    @abstractmethod
    def attribute(self, inputs, targets):
        raise NotImplementedError()