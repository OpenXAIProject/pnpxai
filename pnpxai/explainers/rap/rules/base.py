from typing import Sequence, Union, Optional
from abc import ABC, abstractmethod, abstractproperty
import torch
from torch import Tensor, nn

_TensorOrTensors = Union[Tensor, Sequence[Tensor]]


def safe_divide(a: Tensor, b: Tensor):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())


class RelProp:
    def __init__(self, module: Optional[nn.Module] = None):
        self.module = module

    def gradprop(self, Z: _TensorOrTensors, X: _TensorOrTensors, S: _TensorOrTensors) -> _TensorOrTensors:
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, rel: _TensorOrTensors, inputs: Optional[_TensorOrTensors], outputs: Optional[_TensorOrTensors], args=None, kwargs=None) -> _TensorOrTensors:
        return rel


class RelPropSimple(RelProp):
    def backward(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors) -> _TensorOrTensors:
        Sp = safe_divide(rel, outputs)

        Cp = self.gradprop(outputs, inputs, Sp)[0]
        if torch.is_tensor(inputs):
            inputs = [inputs]

        rel = [datum * Cp for datum in inputs]
        if len(rel) == 1:
            rel = rel[0]

        return rel

    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        if torch.is_tensor(rel):
            rel = [rel]
        rel = [self.backward(datum, inputs, outputs) for datum in rel]
        if len(rel) == 1:
            return rel[0]

        return rel


class RelPropSeparate(RelProp, ABC):
    @abstractproperty
    def agg_dims(self):
        pass

    def shift_rel(self, rel: Tensor, rel_diff: Tensor) -> Tensor:
        rel_nonzero = torch.ne(rel, 0)
        shift = safe_divide(rel_diff, torch.sum(
            rel_nonzero, dim=self.agg_dims, keepdim=True))
        shift = shift * torch.ne(rel, 0)
        rel = rel - shift
        return rel

    @abstractmethod
    def forward(self, weights: Tensor, inputs: Tensor) -> Tensor:
        pass

    def pos_prop(self, rel: Tensor, pos_out: Tensor, neg_out: Tensor, inputs: Tensor):
        rel_pos = torch.clamp(rel, min=0)
        rel_neg = torch.clamp(rel, max=0)
        S1 = safe_divide(rel_pos, pos_out)
        C1 = inputs * self.gradprop(pos_out, inputs, S1)[0]
        S1n = safe_divide(rel_neg, neg_out)
        C1n = inputs * self.gradprop(neg_out, inputs, S1n)[0]
        S2 = safe_divide((rel_pos * safe_divide(neg_out, pos_out + neg_out)), neg_out)
        C2 = inputs * self.gradprop(neg_out, inputs, S2)[0]
        S2n = safe_divide((rel_neg * safe_divide(neg_out, pos_out + neg_out)), neg_out)
        C2n = inputs * self.gradprop(neg_out, inputs, S2n)[0]
        rel_pos = C1 + C2
        rel_neg = C2n + C1n
        new_rel = (rel_pos + rel_neg)
        rel_diff = new_rel.sum(dim=self.agg_dims, keepdim=True) - \
            rel.sum(dim=self.agg_dims, keepdim=True)
        new_rel = self.shift_rel(new_rel, rel_diff)
        return new_rel

    def backward(self, rel: Tensor, pos_x: Tensor, neg_x: Tensor, pos_w: Tensor, neg_w: Tensor):
        rel_nonzero = rel.ne(0).type(rel.type())

        pos_pos_out = self.forward(pos_x, pos_w) * rel_nonzero
        pos_neg_out = -self.forward(pos_x, neg_w) * rel_nonzero
        neg_pos_out = -self.forward(neg_x, pos_w) * rel_nonzero
        neg_neg_out = self.forward(neg_x, neg_w) * rel_nonzero

        pos_rel = self.pos_prop(rel, pos_pos_out, pos_neg_out, pos_x)
        neg_rel = self.pos_prop(rel, neg_pos_out, neg_neg_out, neg_x)
        return pos_rel + neg_rel
