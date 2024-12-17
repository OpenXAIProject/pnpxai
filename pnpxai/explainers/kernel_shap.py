from typing import Tuple, Union, Optional, Dict

from torch import Tensor
from torch.nn.modules import Module
from captum.attr import KernelShap as CaptumKernelShap

from pnpxai.core.detector.types import Linear, Convolution, LSTM, RNN, Attention
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.types import ForwardArgumentExtractor
from pnpxai.explainers.utils.baselines import BaselineMethodOrFunction, BaselineFunction
from pnpxai.explainers.utils.feature_masks import FeatureMaskMethodOrFunction, FeatureMaskFunction
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


class KernelShap(Explainer):
    """
    KernelSHAP explainer.

    Supported Modules: `Linear`, `Convolution`, `LSTM`, `RNN`, `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        n_samples (int): Number of samples
        baseline_fn (Union[BaselineMethodOrFunction, Tuple[BaselineMethodOrFunction]]): The baseline function, accepting the attribution input, and returning the baseline accordingly.
        feature_mask_fn (Union[FeatureMaskMethodOrFunction, Tuple[FeatureMaskMethodOrFunction]): The feature mask function, accepting the attribution input, and returning the feature mask accordingly.
        mask_token_id (Optional[int]): The token id of the mask, used for modalities, utilizing tokenization
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Scott M. Lundberg, Su-In Lee. A Unified Approach to Interpreting Model Predictions.
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
        forward_arg_extractor: Optional[ForwardArgumentExtractor] = None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor] = None,
        mask_token_id: Optional[int] = None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor
        )
        self.n_samples = n_samples
        self.baseline_fn = baseline_fn
        self.feature_mask_fn = feature_mask_fn
        self.mask_token_id = mask_token_id

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
        explainer = CaptumKernelShap(self.model)
        attrs = explainer.attribute(
            inputs=forward_args,
            target=targets,
            baselines=self._get_baselines(forward_args),
            feature_mask=self._get_feature_masks(forward_args),
            n_samples=self.n_samples,
            additional_forward_args=additional_forward_args,
        )
        attrs = format_out_tuple_if_single(attrs)
        return attrs

    def get_tunables(self) -> Dict[str, Tuple[type, Dict]]:
        """
        Provides Tunable parameters for the optimizer

        Tunable parameters:
            `n_samples` (int): Value can be selected in the range of `range(10, 50, 10)`

            `baseline_fn` (callable): BaselineFunction selects suitable values in accordance with the modality

            `feature_mask_fn` (callable): FeatureMaskFunction selects suitable values in accordance with the modality
        """
        return {
            'n_samples': (int, {'low': 10, 'high': 50, 'step': 10}),
            'baseline_fn': (BaselineFunction, {}),
            'feature_mask_fn': (FeatureMaskFunction, {})
        }
