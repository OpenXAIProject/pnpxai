from typing import Optional

import torch
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
from scipy.stats import entropy

from pnpxai.core._types import Model
from pnpxai.explainers.base import Explainer
from pnpxai.evaluator.metrics.base import Metric


class Complexity(Metric):
    """
    Computes the complexity of attributions.

    Given `attributions`, calculates a fractional contribution distribution `prob_mass`,
    ``prob_mass[i] = hist[i] / sum(hist)``. where ``hist[i] = histogram(attributions[i])``.

    The complexity is defined by the entropy,
    ``evaluation = -sum(hist * ln(hist))``


    Args:
        model (Model): The model used for evaluation
        explainer (Optional[Explainer]): The explainer used for evaluation.
        n_bins (int): The number of bins for histogram computation.

    Reference:
        U. Bhatt, A. Weller, and J. M. F. Moura. Evaluating and aggregating feature-based model attributions. In Proceedings of the IJCAI (2020).
    """
    alias = ['complexity']


    def __init__(
        self, model: Model, explainer: Optional[Explainer] = None, n_bins: int = 10
    ):
        super().__init__(model, explainer)
        self.n_bins = n_bins

    def evaluate(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attributions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate the explainer's complexity based on their probability masses.

        Args:
            inputs (Optional[Tensor]): The input tensors to the model.
            targets (Optional[Tensor]): The target labels for the inputs.
            attributions (Optional[Tensor]): The attributions for the inputs.

        Returns:
            Tensor: A tensor of the complexity evaluations.
        """
        attributions = self._get_attributions(inputs, targets, attributions)
        if attributions.ndim == 2: # FIXED
            attributions = attributions.unsqueeze(1) # FIXED

        assert attributions.ndim in [3, 4], "Must have 2D or 3D attributions"
        if attributions.ndim == 4:
            attributions = rgb_to_grayscale(attributions)
        evaluations = []
        for attr in attributions:
            hist, _ = np.histogram(attr.detach().cpu(), bins=self.n_bins)
            prob_mass = hist / hist.sum()
            evaluations.append(entropy(prob_mass))
        return torch.tensor(evaluations).to(attributions.dtype).to(self.device)
