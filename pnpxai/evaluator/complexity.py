from typing import Optional

import torch
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
from scipy.stats import entropy

from pnpxai.core._types import Model
from pnpxai.explainers._explainer import ExplainerWArgs
from pnpxai.evaluator._evaluator import EvaluationMetric

class Complexity(EvaluationMetric):
    def __init__(self, n_bins: int=10):
        self.n_bins = n_bins
                
    def __call__(
            self,
            model: Model,
            explainer_w_args: ExplainerWArgs,
            inputs: torch.Tensor,
            targets: Optional[torch.Tensor]=None,
            attributions: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:
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