from typing import (
	Optional,
	Callable,
	Tuple,
	Union,
	Sequence,
)

from torch import Tensor
from torch.nn.modules import Module

from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.explainers.smooth_grad import SmoothGrad


class VarGrad(SmoothGrad):
	def __init__(
		self,
		model: Module,
		noise_level: float=.1,
		n_iter: int=20,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        layer: Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]=None,
        n_classes: Optional[int]=None,
	) -> None:
		super().__init__(
			model=model,
			noise_level=noise_level,
			n_iter=n_iter,
			forward_arg_extractor=forward_arg_extractor,
			additional_forward_arg_extractor=additional_forward_arg_extractor,
			layer=layer,
			n_classes=n_classes,
		)

	def attribute(
		self,
		inputs: Union[Tensor, Tuple[Tensor]],
		targets: Tensor,
	) -> Union[Tensor, Tuple[Tensor]]:
		forward_args, additional_forward_args = self._extract_forward_args(inputs)
		with self.attributor() as attributor:
			avg_grads, avg_grads_sq = attributor.forward(
				forward_args,
				targets,
				additional_forward_args,
				return_squared=True,
			)
		vargrads = tuple(
			avg_grad_sq - avg_grad for avg_grad_sq, avg_grad in zip(
				format_into_tuple(avg_grads_sq),
				format_into_tuple(avg_grads),
			)
		)
		return format_out_tuple_if_single(vargrads)
