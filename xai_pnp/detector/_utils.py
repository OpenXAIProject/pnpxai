from functools import partial
from typing import Optional, Union, Sequence

import torch

from xai_pnp.core._types import Model
from xai_pnp.detector._core import ModelArchitecture

MULTIPLE_OUTPUT_MODULES = {
    torch.nn.modules.activation.MultiheadAttention,
}

def get_model_architecture(
        model: Model,
        input: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> ModelArchitecture:
    model_architecture = ModelArchitecture()

    module_mode = model.training
    model.eval()
    def _record_layer_info(
        module: Model,
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
            output_size = tuple(output.size()),
            grad_fn = output.grad_fn.name(),
        ))
    hook_handles = []
    for n, m in model.named_modules():
         # if a layer is not named, the layer is not recorded
        hook_handles.append(
            m.register_forward_hook(
                partial(_record_layer_info, name=n)
            )
        )
    
    _ = model(input)
    for handle in hook_handles:
        handle.remove()
    
    model.training = module_mode
    return model_architecture
    

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
input = torch.randn(1, 3, 224, 224)
model_architecture = get_model_architecture(model, input)
for layer_info in model_architecture.data:
    print(layer_info.name, layer_info.grad_fn)