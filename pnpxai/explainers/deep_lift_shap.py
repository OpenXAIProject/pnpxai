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
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(model)
        self.background_data = background_data
        self.dtype = dtype or torch.float64

    def attribute(self, inputs: Tensor, targets: Tensor) -> Tensor:
        _explainer = shap.DeepExplainer(self.model, self.background_data)
        shap_values = _explainer.shap_values(inputs)
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][0]
        return torch.tensor(
            shap_values[targets.cpu().item()],
            dtype=self.dtype,
            device=self.device,
        )