from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
from typing import Union, Sequence, Dict, Any, Literal
from enum import Enum

TensorSequence = Union[Dataset, Sequence[Tensor]]
TensorOrTensorSequence = Union[TensorSequence, Tensor]
DataSource = Union[DataLoader, TensorOrTensorSequence]

Model = nn.Module  # TODO: List other model types in Union[Type1, Type2, ...]
Task = Literal["image", "tabular"]
Question = Literal["why", "how"]
Modality = Literal['image', 'text']
ExplanationType = Literal['attribution']


@dataclass
class Args:
    args: Sequence[Any]
    kwargs: Dict[str, Any]

    def __post_init__(self):
        self.args = self.args or []
        self.kwargs = self.kwargs or {}


class ConfigKeys(Enum):
    EXPLAINERS = 'explainers'
    METRICS = 'metrics'
    TASK = 'task'
