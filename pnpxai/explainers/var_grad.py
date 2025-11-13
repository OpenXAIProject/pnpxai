from typing import Optional, Callable, Tuple, Union, Any, List

from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single
from pnpxai.explainers.smooth_grad import SmoothGrad
from pnpxai.explainers.types import TargetLayerOrTupleOfTargetLayers


class VarGrad(SmoothGrad):
    """
    VarGrad explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        noise_level (float): The noise level added during attribution.
        n_iter (int): The number of iterations, the input is modified.
        target_layer (Optional[Union[Union[str, Module], Sequence[Union[str, Module]]]]): The target module to be explained.
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).       
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Lorenz Richter, Ayman Boustati, Nikolas Nüsken, Francisco J. R. Ruiz, Ömer Deniz Akyildiz. VarGrad: A Low-Variance Gradient Estimator for Variational Inference.
    """
    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]
    SUPPORTED_DTYPES = [float, int]
    SUPPORTED_NDIMS = [2, 4]
    alias = ['var_grad', 'vg']
    
    def __init__(
        self,
        model: Module,
        noise_level: float = .1,
        n_iter: int = 20,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
        target_layer: Optional[TargetLayerOrTupleOfTargetLayers] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            noise_level=noise_level,
            n_iter=n_iter,
            target_input_keys=target_input_keys,
            additional_input_keys=additional_input_keys,
            output_modifier=output_modifier,
            target_layer=target_layer,
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
        forward_args, additional_forward_args = self.format_inputs(inputs)
        with self.attributor() as attributor:
            avg_grads, avg_grads_sq = attributor.forward(
                format_out_tuple_if_single(forward_args),
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
