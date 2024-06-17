from typing import Callable, Tuple, Union, Optional

import torch
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import KernelShap as CaptumKernelShap

from .base import Explainer
from .utils import default_feature_mask_func


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
        self.feature_mask_fn = feature_mask_fn or default_feature_mask_func


    def attribute(
        self,
        inputs: Tensor,
        targets: Optional[Tensor]=None,
    ) -> Union[Tensor, Tuple[Tensor]]:
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        baselines = self.baseline_fn(forward_args)
        feature_mask = self.feature_mask_fn(forward_args)
        
        explainer = CaptumKernelShap(self.model)
        attrs = explainer.attribute(
            inputs=forward_args,
            target=targets,
            baselines=baselines,
            feature_mask=feature_mask,
            n_samples=self.n_samples,
            additional_forward_args=additional_forward_args,
        )
        return attrs