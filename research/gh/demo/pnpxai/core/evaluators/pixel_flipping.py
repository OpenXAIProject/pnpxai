from torch import Tensor, rand
from torch.nn import Module
from quantus import PixelFlipping as QuantusPixelFlipping

from .base import EvaluatorInterface

def get_factors(number):
    n = int(number**.5) + 1
    x = number
    divisors = []
    era = [1] * n
    primes = []
    for p in range(2, n):
        if era[p]:
            primes.append(p)
            while x % p == 0:
                x //= p
                divisors.append(p)
            for i in range(p*p, n, p):
                era[i] = False
    if x != 1:
        divisors.append(x)
    return divisors


class PixelFlipping(EvaluatorInterface):
	RELATED_PROPERTIES = {'correctness', 'output_completeness', 'compactness'}
	
	def __init__(
		self,
		model: Module,
		n_features: int,
		min_steps: int = 10,
		max_steps: int = 100,
		**metric_args
	):
		self.n_features = n_features
		self.min_steps = min_steps
		self.max_steps = max_steps
		metric_args['features_in_step'] = self._get_features_in_step()
		super().__init__(
			model = model,
			metric = QuantusPixelFlipping(**metric_args),
			related_properties = self.RELATED_PROPERTIES,
		)
	
	def _get_features_in_step(self) -> int:
		factors = get_factors(self.n_features)
		n_steps = factors.pop(0)
		while factors:
			factor = factors.pop(0)
			_n_steps = n_steps * factor
			if (_n_steps >= self.min_steps) and (_n_steps <= self.max_steps):
				break
			n_steps = _n_steps
		return self.n_features // n_steps

	def random_flipping(self, inputs: Tensor, targets: Tensor) -> Tensor:
		return self.evaluate(
			inputs = inputs,
			targets = targets,
			attributions = rand(inputs.shape),
		)
