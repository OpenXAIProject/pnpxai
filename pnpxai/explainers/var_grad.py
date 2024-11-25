from typing import (
	Optional,
	Callable,
	Tuple,
	Union,
	Sequence,
)

from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.explainers.smooth_grad import SmoothGrad


class VarGrad(SmoothGrad):
	"""
    VarGrad explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
		noise_level (float): The noise level added during attribution.
		n_iter (int): The number of iterations, the input is modified.
		layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained.
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Lorenz Richter, Ayman Boustati, Nikolas Nüsken, Francisco J. R. Ruiz, Ömer Deniz Akyildiz. VarGrad: A Low-Variance Gradient Estimator for Variational Inference.
    """

	SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]

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
		"""
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            torch.Tensor: The result of the explanation.
        """
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
