from torch.utils.data import DataLoader, Dataset
from torch import Tensor, nn
from typing import Optional, Union, Sequence

TensorSequence = Union[DataLoader, Dataset, Sequence[Tensor]]
DataSource = Union[TensorSequence, Tensor]

Model = nn.Module # TODO: List other model types in Union[Type1, Type2, ...]