from typing import Callable, Tuple, Union, Optional, List

import torch
import pandas as pd
import numpy as np
from torch import Tensor
from torch.nn.modules import Module
from captum.attr import Lime as CaptumLime
from lime.lime_tabular import LimeTabularExplainer
from pnpxai.explainers.sklearn.base import SklearnExplainer
from pnpxai.explainers.sklearn.utils import format_into_array, iterate_inputs

from pnpxai.utils import format_into_tuple
from .base import Explainer


class Lime(Explainer):
    def __init__(
            self,
            model: Module,
            n_samples: int=25,
            baseline_fn: Optional[Callable[[Tensor], Union[Tensor, float]]]=None,
            feature_mask_fn: Optional[Callable[[Tensor], Tensor]]=None,
            perturb_fn: Optional[Callable[[Tensor], Tensor]]=None,
            forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
            additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Tuple[Tensor]]]= None,
    ) -> None:
        super().__init__(model, forward_arg_extractor, additional_forward_arg_extractor)
        self.baseline_fn = baseline_fn or torch.zeros_like
        self.feature_mask_fn = feature_mask_fn
        self.perturb_fn = perturb_fn
        self.n_samples = n_samples
    
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
        
        explainer = CaptumLime(self.model, perturb_func=self.perturb_fn)
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


class TabLime(SklearnExplainer):
    def __init__(
        self,
        model: Callable,
        background_data: np.ndarray,
        categorical_features: Optional[List[int]]=None,
        n_samples: Optional[int]=1000,
    ):
        super().__init__(model, format_into_array(background_data))
        self.categorical_features = format_into_array(categorical_features)
        self.n_samples = n_samples

    def attribute(
        self,
        inputs: np.array,
        targets: Optional[np.array]=None,
    ) -> List[np.ndarray]:
        explainer = LimeTabularExplainer(
            self.background_data,
            categorical_features=self.categorical_features, 
            verbose=False, 
            mode=self.mode,
        )
        
        attrs = []
        for loc, (_, inp) in enumerate(iterate_inputs(inputs)):
            inp = format_into_array(inp)
            label = targets[loc] if self.mode == 'classification' else 1
            res = explainer.explain_instance(
                data_row=inp,
                labels=format_into_tuple(label),
                predict_fn=self._predict_fn,
                num_samples=self.n_samples,
                num_features=len(inp),
            )
            attrs.append(format_into_array([tp[1] for tp in res.as_map()[label]]))
        attrs = format_into_array(attrs)
        if isinstance(inputs, pd.DataFrame):
            return pd.DataFrame(index=inputs.index, columns=inputs.columns, data=attrs)
        return attrs
