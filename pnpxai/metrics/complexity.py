from typing import Dict, Optional, Callable

import torch
from torch import nn
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
from scipy.stats import entropy

from .base import Metric


class Complexity(Metric):
    def __init__(self, model: nn.Module, n_bins: int = 10):
        super().__init__(model)
        self.n_bins = n_bins
    
    def evaluate(
        self,
        inputs: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        attributions: torch.Tensor,
        explain_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        assert attributions.ndim in [3, 4], "Must have 2D or 3D attributions"
        if attributions.ndim == 4:
            attributions = rgb_to_grayscale(attributions)
        evaluations = []
        for attr in attributions:
            hist, _ = np.histogram(attr.detach().cpu(), bins=self.n_bins)
            prob_mass = hist / hist.sum()
            evaluations.append(entropy(prob_mass))
        return torch.tensor(evaluations)


# def pnpxai_complexity(
#         attributions: torch.Tensor,
#         n_bins: int=10,
#         **kwargs
#     ) -> torch.Tensor:
#     """
#     Computes the complexity of attributions.
    
#     Given `attributions`, calculates a fractional contribution distribution `prob_mass`,
#     ``prob_mass[i] = hist[i] / sum(hist)``. where ``hist[i] = histogram(attributions[i])``.

#     The complexity is defined by the entropy,
#     ``evaluation = -sum(hist * ln(hist))``
    
    
#     Args:
#         n_bins (int): The number of bins for histogram computation.
#         attributions (torch.Tensor): The attributions of the inputs.

#     Reference:
#         U. Bhatt, A. Weller, and J. M. F. Moura. Evaluating and aggregating feature-based model attributions. In Proceedings of the IJCAI (2020).
#     """
#     assert attributions.ndim in [3, 4], "Must have 2D or 3D attributions"
#     if attributions.ndim == 4:
#         attributions = rgb_to_grayscale(attributions)
#     evaluations = []
#     for attr in attributions:
#         hist, _ = np.histogram(attr.detach().cpu(), bins=n_bins)
#         prob_mass = hist / hist.sum()
#         evaluations.append(entropy(prob_mass))
#     return torch.tensor(evaluations)