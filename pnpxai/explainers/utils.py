from typing import Sequence, Any, Union

import torch
from torch import nn
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from skimage.segmentation import felzenszwalb

from pnpxai.core.detector import symbolic_trace
from pnpxai.core.detector.utils import get_target_module_of, find_nearest_user_of
from pnpxai.core.detector.types import Convolution, Pool


def find_cam_target_layer(model: nn.Module) -> nn.Module:
    traced_model = symbolic_trace(model)
    last_conv_node = next(
        node for node in reversed(traced_model.graph.nodes)
        if isinstance(get_target_module_of(node), Convolution)
    )
    if last_conv_node is None:
        return
    pool_user = find_nearest_user_of(last_conv_node, Pool)
    if pool_user is None:
        return get_target_module_of(last_conv_node)
    conv_module_nm = next(iter(pool_user.prev.meta.get("nn_module_stack")))
    target_module = model
    for t in conv_module_nm.split("."):
        target_module = getattr(target_module, t)
    return target_module


def default_feature_mask_func(inputs: torch.Tensor, scale=250):
    feature_mask = [
        torch.tensor(felzenszwalb(input.permute(1,2,0).detach().cpu().numpy(), scale=scale))
        for input in inputs
    ]
    return torch.LongTensor(torch.stack(feature_mask)).to(inputs.device)


def captum_wrap_model_input(model):
    if isinstance(model, nn.DataParallel):
        return ModelInputWrapper(model.module)
    return ModelInputWrapper(model)



def _format_to_tuple(obj: Union[Any, Sequence[Any]]):
    if isinstance(obj, Sequence):
        return tuple(obj)
    return (obj,)