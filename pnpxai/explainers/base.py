import abc
from abc import abstractmethod
import sys
from typing import Tuple, Optional, Union, Type, Dict
import math

import copy
from torch import Tensor
from torch.nn.modules import Module

from pnpxai.core._types import ExplanationType
from pnpxai.explainers.types import ForwardArgumentExtractor
from pnpxai.explainers.utils import UtilFunction, BaselineFunction, FeatureMaskFunction
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single


# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str("ABC"), (), {})


NON_DISPLAYED_ATTRS = [
    "model",
    "forward_arg_extractor",
    "additional_forward_arg_extractor",
    "device",
    "n_classes",
    "zennit_composite",
]


class Explainer(ABC):
    """
    Abstract base class for implementing attribution explanations for machine learning models.

    This class provides methods for extracting forward arguments, loading baseline and feature mask functions, and applying them during attribution.

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
        **kwargs: Additional keyword arguments to pass to the constructor.

    Notes:
        - Subclasses must implement the `attribute` method to define how attributions are computed.
        - The `forward_arg_extractor` and `additional_forward_arg_extractor` functions allow for customization in extracting forward arguments from the inputs.
    """

    EXPLANATION_TYPE: ExplanationType = "attribution"
    SUPPORTED_MODULES = []
    TUNABLES = {}

    def __init__(
        self,
        model: Module,
        forward_arg_extractor: Optional[ForwardArgumentExtractor] = None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor] = None,
        **kwargs,
    ) -> None:
        self.model = model.eval()
        self.forward_arg_extractor = forward_arg_extractor
        self.additional_forward_arg_extractor = additional_forward_arg_extractor

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __repr__(self):
        kwargs_repr = ', '.join(
            '{}={}'.format(key, value)
            for key, value in self.__dict__.items()
            if key not in NON_DISPLAYED_ATTRS and value is not None
        )
        return "{}({})".format(self.__class__.__name__, kwargs_repr)

    def _extract_forward_args(
        self, inputs: Union[Tensor, Tuple[Tensor]]
    ) -> Tuple[Union[Tensor, Tuple[Tensor], Type[None]]]:
        forward_args = (
            self.forward_arg_extractor(inputs) if self.forward_arg_extractor else inputs
        )
        additional_forward_args = (
            self.additional_forward_arg_extractor(inputs)
            if self.additional_forward_arg_extractor
            else None
        )
        return forward_args, additional_forward_args

    def copy(self):
        return copy.copy(self)

    def set_kwargs(self, **kwargs):
        clone = self.copy()
        for k, v in kwargs.items():
            setattr(clone, k, v)
        return clone

    def _load_util_fn(
        self, util_attr: str, util_fn_class: Type[UtilFunction]
    ) -> Optional[Union[UtilFunction, Tuple[UtilFunction]]]:
        attr = getattr(self, util_attr)
        if attr is None:
            return None

        attr_values = []
        for attr_value in format_into_tuple(attr):
            if isinstance(attr_value, str):
                attr_value = util_fn_class.from_method(method=attr_value)
            attr_values.append(attr_value)
        attr_values = tuple(attr_values)
        return format_out_tuple_if_single(attr_values)

    def _get_baselines(self, forward_args) -> Union[Tensor, Tuple[Tensor]]:
        baseline_fns = self._load_util_fn("baseline_fn", BaselineFunction)
        if baseline_fns is None:
            return None

        forward_args = format_into_tuple(forward_args)
        baseline_fns = format_into_tuple(baseline_fns)

        assert len(forward_args) == len(baseline_fns)
        baselines = tuple(
            baseline_fn(forward_arg)
            for baseline_fn, forward_arg in zip(baseline_fns, forward_args)
        )
        return format_out_tuple_if_single(baselines)

    def _get_feature_masks(self, forward_args) -> Union[Tensor, Tuple[Tensor]]:
        feature_mask_fns = self._load_util_fn("feature_mask_fn", FeatureMaskFunction)
        if feature_mask_fns is None:
            return None
        
        feature_mask_fns = format_into_tuple(feature_mask_fns)
        forward_args = format_into_tuple(forward_args)

        assert len(forward_args) == len(feature_mask_fns)
        feature_masks = []
        max_vals = None
        for feature_mask_fn, forward_arg in zip(feature_mask_fns, forward_args):
            feature_mask = feature_mask_fn(forward_arg)
            if max_vals is not None:
                feature_mask += (
                    max_vals[(...,) + (None,) * (feature_mask.dim() - 1)] + 1
                )
            feature_masks.append(feature_mask)

            # update max_vals
            bsz, *size = feature_mask.size()
            max_vals = feature_mask.view(-1, math.prod(size)).max(axis=1).values
        feature_masks = tuple(feature_masks)
        return format_out_tuple_if_single(feature_masks)

    @abstractmethod
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (Union[Tensor, Tuple[Tensor]]): The inputs for the model.
            targets (Tensor): The target labels.

        Returns:
            Union[Tensor, Tuple[Tensor]]: The computed attributions.
        """
        raise NotImplementedError

    def get_tunables(self) -> Dict[str, Tuple[type, dict]]:
        """
        Returns a dictionary of tunable parameters for the explainer.

        Returns:
            Dict[str, Tuple[type, dict]]: Dictionary of tunable parameters.
        """
        return {}
