from typing import List

from torch import Tensor
from torch.nn import Module
from quantus.metrics.base import Metric

class EvaluatorInterface:
	def __init__(
		self,
		model: Module,
		metric: Metric,
		related_properties: List[str],
	):
	
		self.model = model
		self.metric = metric
		self.related_properties = related_properties
	
	def evaluate(
		self,
		inputs: Tensor,
		targets: Tensor,
		attributions: Tensor,
	) -> Tensor:
		evaluations = self.metric(
			model = self.model,
			x_batch = inputs.detach().numpy(),
			y_batch = targets.detach().numpy(),
			a_batch = attributions.detach().numpy(),
		)
		return Tensor(evaluations)
