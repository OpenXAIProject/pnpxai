import abc
from abc import abstractmethod
import sys
from typing import Tuple, Callable, Optional, Union, Type
import math

import copy
from torch import Tensor
from torch.nn.modules import Module
from optuna.trial import Trial

from pnpxai.core._types import ExplanationType
from pnpxai.explainers.utils.baselines import BaselineFunction
from pnpxai.explainers.utils.feature_masks import FeatureMaskFunction
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single

# Ensure compatibility with Python 2/3
ABC = abc.ABC if sys.version_info >= (3, 4) else abc.ABCMeta(str('ABC'), (), {})


class Explainer(ABC):
    EXPLANATION_TYPE: ExplanationType = "attribution"

    def __init__(
            self,
            model: Module,
            forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
            additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
            **kwargs
    ) -> None:
        self.model = model.eval()
        self.forward_arg_extractor = forward_arg_extractor
        self.additional_forward_arg_extractor = additional_forward_arg_extractor
        self.device = next(model.parameters()).device

    def _extract_forward_args(
            self,
            inputs: Union[Tensor, Tuple[Tensor]]
    ) -> Tuple[Union[Tensor, Tuple[Tensor], Type[None]]]:
        forward_args = self.forward_arg_extractor(inputs) \
            if self.forward_arg_extractor else inputs
        additional_forward_args = self.additional_forward_arg_extractor(inputs) \
            if self.additional_forward_arg_extractor else None
        return forward_args, additional_forward_args

    def copy(self):
        return copy.copy(self)

    def set_kwargs(self, **kwargs):
        clone = self.copy()
        for k, v in kwargs.items():
            setattr(clone, k, v)
        return clone

    def _load_baseline_fn(self):
        baseline_fns = tuple(
            BaselineFunction(method=baseline_fn)
            if isinstance(baseline_fn, str) else baseline_fn
            for baseline_fn in format_into_tuple(self.baseline_fn)
        )
        return format_out_tuple_if_single(baseline_fns)

    def _get_baselines(self, forward_args):
        forward_args = format_into_tuple(forward_args)
        baseline_fns = format_into_tuple(self._load_baseline_fn())
        assert len(forward_args) == len(baseline_fns)
        baselines = tuple(
            baseline_fn(forward_arg)
            for baseline_fn, forward_arg in zip(baseline_fns, forward_args)
        )
        return format_out_tuple_if_single(baselines)

    def _load_feature_mask_fn(self):
        feature_mask_fns = tuple(
            FeatureMaskFunction(method=feature_mask_fn)
            if isinstance(feature_mask_fn, str) else feature_mask_fn
            for feature_mask_fn in format_into_tuple(self.feature_mask_fn)
        )
        return format_out_tuple_if_single(feature_mask_fns)

    def _get_feature_masks(self, forward_args):
        forward_args = format_into_tuple(forward_args)
        feature_mask_fns = format_into_tuple(self._load_feature_mask_fn())
        assert len(forward_args) == len(feature_mask_fns)
        feature_masks = []
        max_vals = None
        for feature_mask_fn, forward_arg in zip(feature_mask_fns, forward_args):
            feature_mask = feature_mask_fn(forward_arg)
            if max_vals is not None:
                feature_mask += max_vals[(...,)+(None,)*(feature_mask.dim()-1)] + 1
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
        raise NotImplementedError

    def suggest_tunables(
        self,
        trial: Trial,
        key: Optional[str]=None,
    ):
        return {}
