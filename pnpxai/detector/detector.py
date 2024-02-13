from typing import Callable
from dataclasses import dataclass

from torch import nn
import torch.nn.functional as F

from ._core import ModelArchitecture, NodeInfo

@dataclass
class ModelArchitectureSummary:
    has_linear: bool
    has_conv: bool
    has_rnn: bool
    has_transformer: bool

    @property
    def representative(self):
        if (
            self.has_linear
            and not self.has_conv
            and not self.has_rnn
            and not self.has_transformer
        ):
            return "linear"
        elif (
            self.has_conv
            and not self.has_rnn
            and not self.has_transformer
        ):
            return "cnn"
        elif self.has_transformer:
            return "transformer"
        else:
            raise Exception("Cannot determine the representative of model architecture.")


def detect_model_architecture(model) -> ModelArchitectureSummary:
    ma = ModelArchitecture(model)
    return ModelArchitectureSummary(
        has_linear=_has_linear(ma),
        has_conv=_has_conv(ma),
        has_rnn=_has_rnn(ma),
        has_transformer=_has_transformer(ma),
    )


def _is_module_of(node: NodeInfo, module: nn.Module):
    return node.opcode == "call_module" and isinstance(node.operator, module)


def _is_function_of(node: NodeInfo, func: Callable):
    return node.opcode == "call_function" and node.operator is func


def _has_linear(ma: ModelArchitecture):
    linear_node = ma.find_node(
        lambda node: (
            _is_module_of(node, nn.Linear)
            or _is_function_of(node, F.linear)
        )
    )
    return linear_node is not None


def _has_conv(ma: ModelArchitecture):
    conv_node = ma.find_node(
        lambda node: (
            _is_module_of(node, nn.Conv1d)
            or _is_module_of(node, nn.Conv2d)
            or _is_function_of(node, F.conv1d)
            or _is_function_of(node, F.conv2d)
        )
    )
    return conv_node is not None


def _has_rnn(ma: ModelArchitecture):
    rnn_node = ma.find_node(
        lambda node: (
            _is_module_of(node, nn.RNN)
        )
    )
    return rnn_node is not None


def _has_transformer(ma: ModelArchitecture):
    transformer_node = ma.find_node(
        lambda node: (
            _is_module_of(node, nn.Transformer)
            or _is_module_of(node, nn.MultiheadAttention)
            or _is_function_of(node, F.scaled_dot_product_attention)
        )
    )
    return transformer_node is not None

