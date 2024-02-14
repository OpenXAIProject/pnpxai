from typing import Optional

import torch
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
from scipy.stats import entropy

from pnpxai.core._types import Model
from pnpxai.explainers._explainer import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric

class Complexity(EvaluationMetric):
    """
    Measures the complexity of attributions.

    Attributes:
        n_bins (int): The number of bins for histogram computation.
    """
    def __init__(self, n_bins: int=10):
        self.n_bins = n_bins
                
    def __call__(
            self,
            attributions: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
        """
        Computes the complexity of attributions.

        Args:
            attributions (torch.Tensor): The attributions of the inputs.
            **kwargs: Additional kwargs to compute metric in an experiment. Not required for single usage.
        """
        assert attributions.ndim in [3, 4], "Must have 2D or 3D attributions"
        if attributions.ndim == 4:
            attributions = rgb_to_grayscale(attributions)
        evaluations = []
        for attr in attributions:
            hist, _ = np.histogram(attr.detach().cpu())
            prob_mass = hist / hist.sum()
            evaluations.append(entropy(prob_mass))
        return torch.tensor(evaluations)