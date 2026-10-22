"""Supported FX operations and their Attribution Equilibrium rules."""

import operator

import torch
from torch import nn
from torch.nn import functional as F

from pnpxai.explainers.ae import rules


SUPPORTED_OPS = {
    "call_module": {
        nn.Conv2d: rules.Conv2d,
        nn.BatchNorm2d: rules.BatchNorm2d,
        nn.ReLU: rules.ReLU,
        nn.Dropout: rules.Dropout,
        nn.Identity: rules.RelProp,
        nn.MaxPool2d: rules.MaxPool2d,
        nn.AvgPool2d: rules.AvgPool2d,
        nn.AdaptiveAvgPool2d: rules.AdaptiveAvgPool2d,
        nn.Flatten: rules.Flatten,
        nn.Linear: rules.Linear,
    },
    "call_function": {
        operator.add: rules.Add,
        operator.getitem: rules.GetItem,
        getattr: rules.GetAttr,
        torch.add: rules.Add,
        torch.cat: rules.Cat,
        torch.flatten: rules.Flatten,
        torch.reshape: rules.Reshape,
        torch.squeeze: rules.Squeeze,
        torch.unsqueeze: rules.Unsqueeze,
        torch.permute: rules.Permute,
        torch.transpose: rules.Transpose,
        torch.relu: rules.ReLU,
        torch.clone: rules.Contiguous,
        F.relu: rules.ReLU,
        F.dropout: rules.Dropout,
        F.max_pool2d: rules.MaxPool2d,
        F.avg_pool2d: rules.AvgPool2d,
        F.adaptive_avg_pool2d: rules.AdaptiveAvgPool2d,
    },
    "call_method": {
        "add": rules.Add,
        "relu": rules.ReLU,
        "relu_": rules.ReLU,
        "flatten": rules.Flatten,
        "reshape": rules.Reshape,
        "view": rules.View,
        "squeeze": rules.Squeeze,
        "unsqueeze": rules.Unsqueeze,
        "permute": rules.Permute,
        "transpose": rules.Transpose,
        "contiguous": rules.Contiguous,
        "clone": rules.Contiguous,
        "detach": rules.Contiguous,
    },
}


def reference_helper_rule(module):
    """Recognize reference AE helper modules without importing that repository."""
    if callable(getattr(module, "hedge", None)):
        return {"Add": rules.Add, "Cat": rules.Cat, "Clone": rules.Clone}.get(
            type(module).__name__
        )
    return None


def module_rule(module):
    """Resolve rules for ordinary PyTorch modules and reference subclasses."""
    helper = reference_helper_rule(module)
    if helper is not None:
        return helper
    for module_type, rule in SUPPORTED_OPS["call_module"].items():
        if isinstance(module, module_type):
            return rule
    return None
