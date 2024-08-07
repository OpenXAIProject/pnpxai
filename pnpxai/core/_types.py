from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
from typing import Union, Sequence, Literal, List, Tuple
from enum import Enum


TensorOrTupleOfTensors = Union[Tensor, Tuple[Tensor]]
TensorSequence = Union[Dataset, Sequence[Tensor]]
TensorOrTensorSequence = Union[TensorSequence, Tensor]
DataSource = Union[DataLoader, TensorOrTensorSequence]

Model = nn.Module  # TODO: List other model types in Union[Type1, Type2, ...]
Task = Literal["classification"]
Modality = Literal["image", "tabular", "time_series", "text"]
ModalityOrListOfModalities = Union[Modality, List[Modality]]
ExplanationType = Literal["attribution"]

class ConfigKeys(Enum):
    EXPLAINERS = 'explainers'
    METRICS = 'metrics'
    TASK = 'task'