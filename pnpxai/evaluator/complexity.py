import torch
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
from scipy.stats import entropy

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
        
        Given `attributions`, calculates a fractional contribution distribution `prob_mass`,
        ``prob_mass[i] = hist[i] / sum(hist)``. where ``hist[i] = histogram(attributions[i])``.

        The complexity is defined by the entropy,
        ``evaluation = -sum(hist * ln(hist))``
        
        Args:
            attributions (torch.Tensor): The attributions of the inputs.
            **kwargs: Additional kwargs to compute metric in an experiment. Not required for single usage.
            
        Reference:
            U. Bhatt, A. Weller, and J. M. F. Moura. Evaluating and aggregating feature-based model explanations. In Proceedings of the IJCAI (2020).
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
    
    def eval_ts(
            self,
            attributions: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
        return self.__call__(attributions=attributions, **kwargs)
        