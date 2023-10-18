from functools import partial
from typing import Optional, Union, Sequence

import torch

from xai_pnp.core._types import Model
from xai_pnp.explainers._explainer import Explainer
from xai_pnp.explainers import IntegratedGradients, RAP

from .core import ModelArchitecture


class Detector:
    def __init__(self):
        pass

    def __call__(self, model: Model) -> Explainer:
        return RAP(model)


def get_model_architecture(
        model, # Model,
        input: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> ModelArchitecture:
    model_architecture = ModelArchitecture()
    module_mode = model.training
    model.eval()
    def _record_layer_info(
        module, #: Model,
        input: torch.Tensor,
        output: torch.Tensor,
        name: Optional[str] = None,
    ) -> None:
        # [TODO] ugly
        if not torch.is_tensor(output):
            if any(isinstance(module, m) for m in MULTIPLE_OUTPUT_MODULES):
                output = output[0]
        model_architecture._add_data(dict(
            name = name,
            module_name = module.__module__,
            class_name = module.__class__.__name__,
            output_size = None, # insert in the second pass
            grad_fn = None, # insert in the second pass
        ))
    hook_handles = []
    for n, m in model.named_modules():
         # if a layer is not named, the layer is not recorded
        hook_handles.append(
            m.register_forward_hook(
                partial(_record_layer_info, name=n)
            )
        )

    if torch.is_tensor(input):
        _ = model(input)
    else:
        _ = model(*input)
    for handle in hook_handles:
        handle.remove()

    model.training = module_mode
    return model_architecture


MULTIPLE_OUTPUT_MODULES = {
    torch.nn.modules.activation.MultiheadAttention,
}