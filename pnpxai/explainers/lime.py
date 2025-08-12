from typing import Callable, Tuple, Union, Optional, Any, List

from torch import Tensor
from torch.nn.modules import Module
from captum.attr import Lime as CaptumLime

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.utils import (
    format_multimodal_supporting_input,
    run_multimodal_supporting_util_fn,
)
from pnpxai.explainers.base import Explainer, Tunable
from pnpxai.explainers.types import TunableParameter
from pnpxai.explainers.utils.types import (
    BaselineFunctionOrTupleOfBaselineFunctions,
    FeatureMaskFunctionOrTupleOfFeatureMaskFunctions,
)
from pnpxai.explainers.utils.baselines import ZeroBaselineFunction
from pnpxai.explainers.utils.feature_masks import Felzenszwalb


class Lime(Explainer, Tunable):
    """
    Lime explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        n_samples (int): Number of samples
        baseline_fn (Union[BaselineMethodOrFunction, Tuple[BaselineMethodOrFunction]]): The baseline function, accepting the attribution input, and returning the baseline accordingly.
        feature_mask_fn (Union[FeatureMaskMethodOrFunction, Tuple[FeatureMaskMethodOrFunction]): The feature mask function, accepting the attribution input, and returning the feature mask accordingly.
        perturb_fn (Optional[Callable[[Tensor], Tensor]]): The perturbation function, accepting the attribution input, and returning the perturbed value.
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin. "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
    """

    SUPPORTED_MODULES = [Linear, Convolution, LSTM, RNN, Attention]
    SUPPORTED_DTYPES = [float, int]
    SUPPORTED_NDIMS = [2, 4]
    alias = ['lime']

    def __init__(
        self,
        model: Module,
        n_samples: int = 25,
        baseline_fn: Optional[BaselineFunctionOrTupleOfBaselineFunctions] = None,
        feature_mask_fn: Optional[FeatureMaskFunctionOrTupleOfFeatureMaskFunctions] = None,
        perturb_fn: Optional[Callable[[Tensor], Tensor]] = None,
        target_input_keys: Optional[List[Union[str, int]]] = None,
        additional_input_keys: Optional[List[Union[str, int]]] = None,
        output_modifier: Optional[Callable[[Any], Tensor]] = None,
    ) -> None:
        self.n_samples = TunableParameter(
            name='n_samples',
            current_value=n_samples,
            dtype=int,
            is_leaf=True,
            space={'low': 10, 'high': 50, 'step': 10},
        )
        baseline_fn = baseline_fn or ZeroBaselineFunction()
        self.baseline_fn = format_multimodal_supporting_input(
            baseline_fn or ZeroBaselineFunction(),
            format=TunableParameter,
            input_key='current_value',
            name='baseline_fn',
            dtype=str,
            is_leaf=False,
        )
        self.feature_mask_fn = format_multimodal_supporting_input(
            feature_mask_fn or Felzenszwalb(),
            format=TunableParameter,
            input_key='current_value',
            name='feature_mask_fn',
            dtype=str,
            is_leaf=False,
        )
        self.perturb_fn = perturb_fn
        Explainer.__init__(
            self,
            model,
            target_input_keys,
            additional_input_keys,
            output_modifier,
        )
        Tunable.__init__(self)
        self.register_tunable_params([
            self.n_samples, self.baseline_fn, self.feature_mask_fn])

    def attribute(
        self,
        inputs: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: The result of the explanation.
        """
        forward_args, additional_forward_args = self.format_inputs(inputs)
        baselines = run_multimodal_supporting_util_fn(forward_args, self.baseline_fn)
        feature_masks = run_multimodal_supporting_util_fn(forward_args, self.feature_mask_fn)

        _explainer = CaptumLime(self._wrapped_model, perturb_func=self.perturb_fn)
        attrs = _explainer.attribute(
            inputs=forward_args,
            target=targets,
            baselines=baselines,
            feature_mask=feature_masks,
            n_samples=self.n_samples.current_value,
            additional_forward_args=additional_forward_args,
        )
        return attrs
    