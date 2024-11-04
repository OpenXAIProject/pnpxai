from typing import Dict, Tuple, Callable

import torch
import shap
import captum
from torch import Tensor
from pnpxai.explainers.base import Explainer
from torch.nn.modules import Module
from captum.attr import DeepLiftShap as CaptumDeepLiftShap
from pnpxai.explainers.base import Explainer
from pnpxai.explainers.types import ForwardArgumentExtractor
from pnpxai.explainers.utils.baselines import BaselineMethodOrFunction, BaselineFunction
from pnpxai.utils import format_into_tuple, format_out_tuple_if_single

'''
[GH] packages: shap v.s. captum

shap package 활용시 (장) 별도 수정 없이 빠르게 구현 가능하나, (단) HPO구현 어려움
captum package 활용시 (장) HPO 등 설명 이후 태스크에 대한 구현 용이, (단) 구현에 시간 좀 걸림

p.s. shap package의 backgound data는 captum의 baseline_fn을 통해 구현할 수 있음
'''


class DeepLiftShapByShap(Explainer):
    def __init__(
        self,
        model: Module,
        background_data: Tensor,
    ):
        super().__init__(model)
        self.background_data = background_data

    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        _explainer = shap.DeepExplainer(self.model, self.background_data)
        shap_values = torch.tensor(_explainer.shap_values(inputs))
        permute_dims = list(range(1, shap_values.dim())) + [0]
        shap_values = shap_values.permute(*permute_dims)
        attrs = shap_values[torch.arange(inputs.size(0)), :, targets.cpu()] # select values for the targeted labels
        attrs = attrs.to(inputs.device).to(inputs.dtype) # match device and dtype to inputs
        return attrs


class DeepLiftShapByCaptum(Explainer):
    def __init__(
        self,
        model: Module,
        baseline_fn: Callable,
    ):
        super().__init__(model)
        self.baseline_fn = baseline_fn

    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        _explainer = captum.attr.DeepLiftShap(model=self.model)
        baselines = self.baseline_fn(inputs)
        attrs = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            target=targets,
        )
        return attrs