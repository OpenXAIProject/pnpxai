from typing import Any, List, Sequence, Tuple
import torch
from torch import Tensor, LongTensor
from captum.attr import KernelShap as KernelShapeCaptum
from captum._utils.typing import BaselineType, TargetType
from plotly import express as px
from plotly.graph_objects import Figure
from skimage.segmentation import felzenszwalb

from pnpxai.core._types import Model, DataSource
from pnpxai.explainers._explainer import Explainer

from typing import Union, Literal


class KernelShap(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = KernelShapeCaptum(model)
        self.device = next(self.model.parameters()).device

    def attribute(
        self,
        inputs: DataSource,
        target: TargetType = None,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False
    ) -> List[Tensor]:
        if feature_mask is None:
            feature_mask = felzenszwalb(
                inputs.permute(0, 2, 3, 1).cpu().numpy()[0], scale=250
            )
            feature_mask = LongTensor(feature_mask).to(self.device)

        attributions = self.method.attribute(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )

        return attributions

    def format_outputs_for_visualization(self, inputs: DataSource, outputs: DataSource, *args, **kwargs) -> Sequence[Figure]:
        return [[
            px.imshow(output.permute((1, 2, 0))) for output in batch
        ] for batch in outputs]
