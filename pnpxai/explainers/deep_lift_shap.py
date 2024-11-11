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


class DeepLiftShap(Explainer):
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
