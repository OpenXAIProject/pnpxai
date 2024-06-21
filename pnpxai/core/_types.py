from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
from typing import Union, Sequence, Literal, Tuple


TensorSequence = Union[Dataset, Sequence[Tensor]]
TensorOrTensorSequence = Union[TensorSequence, Tensor]
DataSource = Union[DataLoader, TensorOrTensorSequence]

Model = nn.Module  # TODO: List other model types in Union[Type1, Type2, ...]
Task = Literal["classification"]
Unimodality = Literal["image", "tabular", "time_series", "text"]
Modality = Union[Unimodality, Tuple[Unimodality]]
Question = Literal["why"]
ExplanationType = Literal["attribution"]