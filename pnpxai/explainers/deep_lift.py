from typing import Callable, Optional
from captum.attr import DeepLift as CaptumDeepLift
import torch
from torch.nn.modules import Module

from .base import Explainer


class DeepLift(Explainer):
    def __init__(
            self,
            model: Module,
            baseline_func: Optional[Callable]=None,
        ) -> None:
        super().__init__(model)
        self.baseline_func = baseline_func or torch.zeros_like
    
    def attribute(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        baselines = self.baseline_func(inputs)
        explainer = DeepLift(self.model)
        attrs = explainer.attribute(
            inputs,
            target=targets,
            baselines=baselines,
        )
        return attrs