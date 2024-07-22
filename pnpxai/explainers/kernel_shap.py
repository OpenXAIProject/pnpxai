from typing import Callable, Tuple, Union, Optional

import torch
import pandas as pd
import numpy as np
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import KernelShap as CaptumKernelShap
from shap import KernelExplainer
from shap.utils._legacy import kmeans

from pnpxai.core._types import Task
from pnpxai.explainers.sklearn.base import SklearnExplainer
from pnpxai.explainers.sklearn.utils import format_into_array
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


def _tab_ks_model_wrapper(predict_fn):
    def callable(inputs):
        return predict_fn(inputs)
    return callable

class TabKernelShap(SklearnExplainer):
    def __init__(
        self,
        model: Callable,
        background_data: pd.DataFrame,
        k_means: int=100,
        n_samples: Optional[int]=None,
    ):
        super().__init__(model, background_data)
        self.n_samples = n_samples if n_samples else 'auto'
        self.k_means = k_means

    def attribute(self, inputs, targets):
        if self.mode == "classification":
            assert len(inputs) == len(targets), "The number of inputs and targets must have same length"

        explainer = KernelExplainer(
            _tab_ks_model_wrapper(self._predict_fn),
            kmeans(self.background_data, self.k_means),
        )
        attrs = explainer.shap_values(
            X=inputs,
            nsamples=self.n_samples,
            show=False,
        )
        if self.mode == 'classification':
            attrs = format_into_array(attrs).transpose(1,2,0)
            attrs = attrs[np.arange(attrs.shape[0]), :, targets]
        if isinstance(inputs, pd.DataFrame):
            attrs = pd.DataFrame(index=inputs.index, columns=inputs.columns, data=attrs)
        return attrs
