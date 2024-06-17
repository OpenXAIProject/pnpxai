import _operator
from typing import Dict, Type
import torch
from torch import nn
from torch.nn import functional as F
from pnpxai.explainers_backup.rap import rules

SUPPORTED_OPS: Dict[str, Dict[str, Type[rules.RelProp]]] = {
    'call_module': {
        nn.ReLU: rules.ReLU,
        nn.GELU: rules.GeLU,
        nn.Dropout: rules.Dropout,
        nn.MaxPool2d: rules.MaxPool2d,
        nn.MaxPool1d: rules.MaxPool1d,
        nn.AdaptiveAvgPool2d: rules.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool1d: rules.AdaptiveAvgPool1d,
        nn.AvgPool2d: rules.AvgPool2d,
        nn.AvgPool1d: rules.AvgPool1d,
        nn.BatchNorm2d: rules.BatchNorm2d,
        nn.BatchNorm1d: rules.BatchNorm1d,
        nn.LayerNorm: rules.LayerNorm,
        nn.Linear: rules.Linear,
        nn.Conv2d: rules.Conv2d,
        nn.Conv1d: rules.Conv1d,
        nn.MultiheadAttention: rules.MultiHeadAttention,
        nn.Flatten: rules.Flatten,
    },
    'call_function': {
        getattr: rules.GetAttr,
        _operator.add: rules.Add,
        _operator.sub: rules.Sub,
        _operator.mul: rules.Mul,
        _operator.floordiv: rules.FloorDiv,
        _operator.getitem: rules.GetItem,
        torch.add: rules.Add,
        torch.flatten: rules.Flatten,
        torch.relu: rules.ReLU,
        torch.cat: rules.Cat,
        torch.transpose: rules.Flatten,
        torch.sub: rules.Sub,
        torch.unsqueeze: rules.Unsqueeze,
        F.avg_pool2d: rules.AvgPool2d,
        F.avg_pool1d: rules.AvgPool1d,
        F.max_pool2d: rules.MaxPool2d,
        F.max_pool1d: rules.MaxPool1d,
        F.relu: rules.ReLU,
        F.gelu: rules.GeLU,
    },
    'call_method': {
        'add': rules.Add,
        'sub': rules.Sub,
        'relu': rules.ReLU,
        'transpose': rules.Flatten,
        'repeat': rules.Repeat,
        'expand': rules.Expand,
        'permute': rules.Permute,
        'reshape': rules.Reshape,
    }
}
