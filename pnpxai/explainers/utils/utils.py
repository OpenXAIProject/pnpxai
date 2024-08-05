from typing import Sequence, Any, Union
import functools

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


def default_feature_mask_fn_image(inputs: torch.Tensor, scale=250):
    feature_mask = [
        torch.tensor(felzenszwalb(input.permute(1,2,0).detach().cpu().numpy(), scale=scale))
        for input in inputs
    ]
    return torch.LongTensor(torch.stack(feature_mask)).to(inputs.device)


def default_feature_mask_fn_text(inputs): return None


def default_feature_mask_fn_image_text(images, text):
    fm_img = default_feature_mask_fn_image(images)
    bsz, text_len = text.size()
    fm_text = torch.arange(text_len).repeat(bsz).view(bsz, text_len)
    fm_text += fm_img.max().item() + 1
    return fm_img, fm_text


def get_default_feature_mask_fn(modality):
    if modality == 'image':
        return default_feature_mask_fn_image
    elif modality == 'text':
        return default_feature_mask_fn_text
    elif modality == ('image', 'text'):
        return default_feature_mask_fn_image_text
    elif modality == 'tabular':
        return default_feature_mask_fn_text
    else:
        raise NotImplementedError(f"default_feature_mask_fn for '{modality}' not supported.")


def default_baseline_fn_image(inputs: torch.Tensor):
    return torch.zeros_like(inputs)

def default_baseline_fn_text(inputs: torch.Tensor, mask_token_id: int=0):
    return torch.ones_like(inputs, dtype=torch.long) * mask_token_id

def default_baseline_fn_image_text(images: torch.Tensor, text: torch.Tensor, mask_token_id: int=0):
    return default_baseline_fn_image(images), default_baseline_fn_text(text, mask_token_id)

def default_baseline_fn_tabular(inputs: torch.Tensor):
    return torch.zeros_like(inputs)

def get_default_baseline_fn(modality, mask_token_id=0):
    if modality == 'image':
        return default_baseline_fn_image
    elif modality == 'text':
        return functools.partial(default_baseline_fn_text, mask_token_id=mask_token_id)
    elif modality == ('image', 'text'):
        return functools.partial(default_baseline_fn_image_text, mask_token_id=mask_token_id)
    elif modality == 'tabular':
        return default_baseline_fn_tabular
    else:
        raise NotImplementedError(f"default_baseline_fn for '{modality}' not supported.")

def captum_wrap_model_input(model):
    if isinstance(model, nn.DataParallel):
        return ModelInputWrapper(model.module)
    return ModelInputWrapper(model)



def _format_to_tuple(obj: Union[Any, Sequence[Any]]):
    if isinstance(obj, Sequence):
        return tuple(obj)
    return (obj,)