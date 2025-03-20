from typing import Optional

import torch
from shap import DeepExplainer
from torch import Tensor
from pnpxai.core.detector.types import Convolution
from pnpxai.explainers.base import Explainer
from torch.nn.modules import Module


class DeepLiftShap(Explainer):
    """
    DeepLiftShap explainer.

    Supported Modules: `Convolution`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        background_data: The background dataset used to train the surrogate model.

    Reference:
        Scott M. Lundberg, Su-In Lee, A Unified Approach to Interpreting Model Predictions
    """
    SUPPORTED_MODULES = [Convolution]
    SUPPORTED_DTYPES = [float]
    SUPPORTED_NDIMS = [2]
    alias = ['deep_lift_shap', 'dls']

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
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            torch.Tensor: The result of the explanation.
        """
        _explainer = DeepExplainer(self.model, self.background_data)
        shap_values = _explainer.shap_values(inputs)
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][0]
        return torch.tensor(
            shap_values[targets.cpu().tolist()],
            dtype=self.dtype,
            device=self.device,
        )
