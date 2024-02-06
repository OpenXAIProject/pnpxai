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
    Measures the complexity of model inputs based on their attributions.

    Attributes:
    - n_bins (int): The number of bins for histogram computation.
    """
    def __init__(self, n_bins: int=10):
        """
        Initializes a Complexity object.

        Args:
        - n_bins (int): The number of bins for histogram computation.
        """
        self.n_bins = n_bins
                
    def __call__(
            self,
            model: Model,
            explainer_w_args: ExplainerWArgs,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor]=None,
            attributions: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:
        """
        Computes the complexity of model inputs.

        Args:
        - model (Model): The model to evaluate.
        - explainer_w_args (ExplainerWArgs): The explainer with arguments.
        - inputs (torch.Tensor): The input data.
        - targets (Optional[torch.Tensor]): The target labels for the inputs (default: None).
        - attributions (Optional[torch.Tensor]): The attributions of the inputs (default: None).

        Returns:
        - torch.Tensor: The complexity evaluations.
        """        
        if attributions is None:
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            outputs = model(inputs)
            if targets is None:
                targets = outputs.argmax(1)
            targets = targets.to(device)
            attributions = explainer_w_args.attribute(inputs, targets, **explainer_w_args.kwargs)

        assert attributions.ndim in [3, 4], "Must have 2D or 3D attributions"
        if attributions.ndim == 4:
            attributions = rgb_to_grayscale(attributions)
        evaluations = []
        for attr in attributions:
            hist, _ = np.histogram(attr.detach().cpu())
            prob_mass = hist / hist.sum()
            evaluations.append(entropy(prob_mass))
        return torch.tensor(evaluations)