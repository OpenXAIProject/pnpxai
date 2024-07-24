from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import Union, Sequence, Literal, List
from enum import Enum

# Data types
TensorSequence = Union[Dataset, Sequence[Tensor]]
TensorOrTensorSequence = Union[TensorSequence, Tensor]
DataSource = Union[DataLoader, TensorOrTensorSequence]

# Model types
TorchModel = nn.Module
XGBModel = Union[XGBClassifier, XGBRegressor]
SklearnModel = Union[ClassifierMixin, RegressorMixin]
Model = Union[TorchModel, XGBModel, SklearnModel]

# Modalities and tasks
Task = Literal['classification', 'regression']
Modality = Literal["image", "tabular", "time_series", "text"]
ModalityOrListOfModalities = Union[Modality, List[Modality]]
Question = Literal["why"]
ExplanationType = Literal["attribution"]

class ConfigKeys(Enum):
    EXPLAINERS = 'explainers'
    METRICS = 'metrics'
    TASK = 'task'