from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class ExplainerInterface(ABC):
	def __init__(
		self,
		model: Module,
	):
		self.model = model
		
	@abstractmethod
	def attribute(
		inputs: Tensor,
		targets: Tensor,
		**tunables,
	) -> Tensor:
		'''
		'''
