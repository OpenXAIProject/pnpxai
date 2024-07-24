from typing import Tuple, Union, Callable, List
from torch import Tensor
from torch.nn.modules import Module

TensorOrTupleOfTensors = Union[Tensor, Tuple[Tensor]]
ForwardArgumentExtractor = Callable[[TensorOrTupleOfTensors], TensorOrTupleOfTensors]
TargetLayer = Union[str, Module]
TargetLayerOrListOfTargetLayers = Union[TargetLayer, List[TargetLayer]]