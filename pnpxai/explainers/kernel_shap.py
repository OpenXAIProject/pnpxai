from typing import Callable, Tuple, Union, Optional

import torch
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import KernelShap as CaptumKernelShap

from pnpxai.utils import format_into_tuple
from .base import Explainer


class KernelShap(Explainer):
    def __init__(
        self,
        model: Module,
        n_samples: int=25,
        baseline_fn: Optional[Callable[[Tensor], Union[Tensor, float]]]=None,
        feature_mask_fn: Optional[Callable[[Tensor], Tensor]]=None,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor
        )
        self.n_samples = n_samples
        self.baseline_fn = baseline_fn or torch.zeros_like
        self.feature_mask_fn = feature_mask_fn


    def attribute(
        self,
        inputs: Tensor,
        targets: Optional[Tensor]=None,
    ) -> Union[Tensor, Tuple[Tensor]]:
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        forward_args = format_into_tuple(forward_args)
        baselines = self.baseline_fn(*forward_args)
        baselines = format_into_tuple(baselines)
        feature_mask = self.feature_mask_fn(*forward_args)
        
        explainer = CaptumKernelShap(self.model)
        attrs = explainer.attribute(
            inputs=forward_args,
            target=targets,
            baselines=baselines,
            feature_mask=feature_mask,
            n_samples=self.n_samples,
            additional_forward_args=additional_forward_args,
        )
        if isinstance(attrs, tuple) and len(attrs) == 1:
            attrs = attrs[0]
        return attrs