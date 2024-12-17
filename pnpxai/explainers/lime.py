from typing import Callable, Tuple, Union, Optional, Dict, Any

import torch
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import Lime as CaptumLime

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.utils.baselines import BaselineMethodOrFunction, BaselineFunction
from pnpxai.explainers.utils.feature_masks import FeatureMaskMethodOrFunction, FeatureMaskFunction
from pnpxai.utils import format_into_tuple


class Lime(Explainer):
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

    def __init__(
        self,
        model: Module,
        n_samples: int = 25,
        baseline_fn: Union[BaselineMethodOrFunction,
                           Tuple[BaselineMethodOrFunction]] = 'zeros',
        feature_mask_fn: Union[FeatureMaskMethodOrFunction,
                               Tuple[FeatureMaskMethodOrFunction]] = 'felzenszwalb',
        perturb_fn: Optional[Callable[[Tensor], Tensor]] = None,
        forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]] = None,
        additional_forward_arg_extractor: Optional[Callable[[
            Tuple[Tensor]], Tuple[Tensor]]] = None,
    ) -> None:
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.baseline_fn = baseline_fn or torch.zeros_like
        self.feature_mask_fn = feature_mask_fn
        self.perturb_fn = perturb_fn
        self.n_samples = n_samples

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
        forward_args, additional_forward_args = self._extract_forward_args(
            inputs)
        forward_args = format_into_tuple(forward_args)

        explainer = CaptumLime(self.model, perturb_func=self.perturb_fn)
        attrs = explainer.attribute(
            inputs=forward_args,
            target=targets,
            baselines=self._get_baselines(forward_args),
            feature_mask=self._get_feature_masks(forward_args),
            n_samples=self.n_samples,
            additional_forward_args=additional_forward_args,
        )
        if isinstance(attrs, tuple) and len(attrs) == 1:
            attrs = attrs[0]
        return attrs

    
    def get_tunables(self) -> Dict[str, Tuple[type, Dict]]:
        """
        Provides Tunable parameters for the optimizer

        Tunable parameters:
            `n_samples` (int): Value can be selected in the range of `range(10, 100, 10)`

            `baseline_fn` (callable): BaselineFunction selects suitable values in accordance with the modality

            `feature_mask_fn` (callable): FeatureMaskFunction selects suitable values in accordance with the modality
        """
        return {
            'n_samples': (int, {'low': 10, 'high': 100, 'step': 10}),
            'baseline_fn': (BaselineFunction, {}),
            'feature_mask_fn': (FeatureMaskFunction, {})
        }
