import torch
import torch.nn as nn
from torch import nn, Tensor
from typing import Sequence, Union, Optional
import torch.nn.functional as F


__all__ = ['Cat', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide']

_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]


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


class ReLU(RelProp):
    pass


class GeLU(RelProp):
    pass


class Dropout(RelProp):
    pass


class MaxPool2d(RelPropSimple):
    pass


class AdaptiveAvgPool2d(RelPropSimple):
    pass


class AvgPool1d(RelPropSimple):
    pass


class AvgPool2d(RelPropSimple):
    pass


class Add(RelPropSimple):
    def _partial_relprop(self, rel: Tensor, inputs: _TensorOrTensors, is_pos: bool, args=None, kwargs=None):
        clamp_kwargs = {'min': 0} if is_pos else {'max': 0}
        inputs = [torch.clamp(x, **clamp_kwargs) for x in inputs]
        pos_outputs = torch.add(*inputs)
        return super().relprop(rel, inputs, pos_outputs, args, kwargs)

    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        if torch.is_tensor(inputs):
            return rel

        pos_rel = self._partial_relprop(rel, inputs, True, args, kwargs)
        neg_rel = self._partial_relprop(rel, inputs, False, args, kwargs)

        if torch.is_tensor(pos_rel):
            return pos_rel + neg_rel
        return [p + n for p, n in zip(pos_rel, neg_rel)]


class Sub(Add):
    pass


class Mul(RelPropSimple):
    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        if torch.is_tensor(inputs):
            inputs = [inputs]

        inputs = [val for val in inputs if torch.is_tensor(val)]
        if len(inputs) <= 1:
            return rel

        return super().relprop(rel, inputs, outputs, args, kwargs)


class FloorDiv(Mul):
    pass


class Flatten(RelProp):
    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        return rel.reshape(inputs.shape)


class Cat(RelPropSimple):
    def backward(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        Sp = safe_divide(rel, outputs)
        Cp = self.gradprop(outputs, inputs, Sp)
        rel = [x * cp for x, cp in zip(inputs, Cp)]

        return rel


class BatchNorm2d(RelProp):
    pass


class LayerNorm(RelProp):
    pass


class Linear(RelProp):
    def shift_rel(self, R, R_val):
        R_nonzero = torch.ne(R, 0).type(R.type())
        shift = safe_divide(R_val, torch.sum(
            R_nonzero, dim=-1, keepdim=True)) * torch.ne(R, 0).type(R.type())
        K = R - shift
        return K

    def pos_prop(self, R: Tensor, Za1: Tensor, Za2: Tensor, x1: Tensor):
        R_pos = torch.clamp(R, min=0)
        R_neg = torch.clamp(R, max=0)
        S1 = safe_divide(
            (R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
        C1 = x1 * self.gradprop(Za1, x1, S1)[0]
        S1n = safe_divide(
            (R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
        C1n = x1 * self.gradprop(Za1, x1, S1n)[0]
        S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
        C2 = x1 * self.gradprop(Za2, x1, S2)[0]
        S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
        C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
        Cp = C1 + C2
        Cn = C2n + C1n

        C = (Cp + Cn)
        C = self.shift_rel(C, C.sum(dim=-1, keepdim=True) -
                           R.sum(dim=-1, keepdim=True))
        return C

    def f(self, R, w1, w2, x1, x2):
        R_nonzero = R.ne(0).type(R.type())
        Za1 = F.linear(x1, w1) * R_nonzero
        Za2 = - F.linear(x1, w2) * R_nonzero

        Zb1 = - F.linear(x2, w1) * R_nonzero
        Zb2 = F.linear(x2, w2) * R_nonzero

        C1 = self.pos_prop(R, Za1, Za2, x1)
        C2 = self.pos_prop(R, Zb1, Zb2, x2)

        return C1 + C2

    def first_prop(self, pd, px, nx, pw, nw):
        Rpp = F.linear(px, pw) * pd
        Rpn = F.linear(px, nw) * pd
        Rnp = F.linear(nx, pw) * pd
        Rnn = F.linear(nx, nw) * pd
        Pos = (Rpp + Rnn).sum(dim=-1, keepdim=True)
        Neg = (Rpn + Rnp).sum(dim=-1, keepdim=True)

        Z1 = F.linear(px, pw)
        Z2 = F.linear(px, nw)
        Z3 = F.linear(nx, pw)
        Z4 = F.linear(nx, nw)

        S1 = safe_divide(Rpp, Z1)
        S2 = safe_divide(Rpn, Z2)
        S3 = safe_divide(Rnp, Z3)
        S4 = safe_divide(Rnn, Z4)
        C1 = px * self.gradprop(Z1, px, S1)[0]
        C2 = px * self.gradprop(Z2, px, S2)[0]
        C3 = nx * self.gradprop(Z3, nx, S3)[0]
        C4 = nx * self.gradprop(Z4, nx, S4)[0]
        bp = self.module.bias * pd * safe_divide(Pos, Pos + Neg)
        bn = self.module.bias * pd * safe_divide(Neg, Pos + Neg)
        Sb1 = safe_divide(bp, Z1)
        Sb2 = safe_divide(bn, Z2)
        Cb1 = px * self.gradprop(Z1, px, Sb1)[0]
        Cb2 = px * self.gradprop(Z2, px, Sb2)[0]
        return C1 + C4 + Cb1 + C2 + C3 + Cb2

    def backward(self, R_p, px, nx, pw, nw):
        Rp = self.f(R_p, pw, nw, px, nx)
        return Rp

    def redistribute(self, Rp_tmp):
        Rp = torch.clamp(Rp_tmp, min=0)
        Rn = torch.clamp(Rp_tmp, max=0)
        R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
        Rp_tmp3 = safe_divide(Rp, R_tot) * \
            (Rp + Rn).sum(dim=-1, keepdim=True)
        Rn_tmp3 = -safe_divide(Rn, R_tot) * \
            (Rp + Rn).sum(dim=-1, keepdim=True)
        return Rp_tmp3 + Rn_tmp3

    def relprop(self, R_p, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        pw = torch.clamp(self.module.weight, min=0)
        nw = torch.clamp(self.module.weight, max=0)
        X = inputs
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) and R_p.max() == 1:  # first propagation
            Rp_tmp = self.first_prop(R_p, px, nx, pw, nw)
            A = self.redistribute(Rp_tmp)

            return A
        else:
            Rp = self.backward(R_p, px, nx, pw, nw)

        return Rp


class Conv2d(RelProp):
    def _get_conv_output(self, x: Tensor, w: Tensor):
        return torch.conv2d(x, w, stride=self.module.stride, padding=self.module.padding)

    def gradprop2(self, inputs, outputs, DY, weight):
        Z = outputs

        output_padding = inputs.size(2) - (
            (Z.size(2) - 1) * self.module.stride[0] - 2 * self.module.padding[0] + self.module.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.module.stride, padding=self.module.padding, output_padding=output_padding)

    def shift_rel(self, R, R_val):
        R_nonzero = torch.ne(R, 0).type(R.type())
        shift = safe_divide(R_val, torch.sum(
            R_nonzero, dim=[1, 2, 3], keepdim=True)) * torch.ne(R, 0).type(R.type())
        K = R - shift
        return K

    def pos_prop(self, rel: Tensor, Za1: Tensor, Za2: Tensor, inputs: Tensor):
        rel_pos = torch.clamp(rel, min=0)
        rel_neg = torch.clamp(rel, max=0)
        S1 = safe_divide(rel_pos, Za1)
        C1 = inputs * self.gradprop(Za1, inputs, S1)[0]
        S1n = safe_divide(rel_neg, Za2)
        C1n = inputs * self.gradprop(Za2, inputs, S1n)[0]
        S2 = safe_divide((rel_pos * safe_divide((Za2), Za1 + Za2)), Za2)
        C2 = inputs * self.gradprop(Za2, inputs, S2)[0]
        S2n = safe_divide((rel_neg * safe_divide((Za2), Za1 + Za2)), Za2)
        C2n = inputs * self.gradprop(Za2, inputs, S2n)[0]
        Cp = C1 + C2
        Cn = C2n + C1n
        new_rel = (Cp + Cn)
        agg_dims = list(range(1, inputs.ndim))
        rel_diff = new_rel.sum(dim=agg_dims, keepdim=True) - \
            rel.sum(dim=agg_dims, keepdim=True)
        new_rel = self.shift_rel(new_rel, rel_diff)
        return new_rel

    def backward(self, rel: Tensor, pos_x: Tensor, neg_x: Tensor, pos_w: Tensor, neg_w: Tensor):
        rel_nonzero = rel.ne(0).type(rel.type())

        pos_pos_out = self._get_conv_output(pos_x, pos_w) * rel_nonzero
        pos_neg_out = -self._get_conv_output(pos_x, neg_w) * rel_nonzero
        neg_pos_out = -self._get_conv_output(neg_x, pos_w) * rel_nonzero
        neg_neg_out = self._get_conv_output(neg_x, neg_w) * rel_nonzero

        C1 = self.pos_prop(rel, pos_pos_out, pos_neg_out, pos_x)
        C2 = self.pos_prop(rel, neg_pos_out, neg_neg_out, neg_x)
        return C1 + C2

    def final_backward(self, rel: Tensor, inputs: Tensor, outputs: Tensor, pos_w: Tensor, neg_w: Tensor):
        agg_dims = tuple(range(1, inputs.ndim))
        template = torch.zeros_like(inputs)
        low = template + torch.amin(inputs, dim=agg_dims, keepdim=True)
        high = template + torch.amax(inputs, dim=agg_dims, keepdim=True)

        norm_out = self._get_conv_output(inputs, self.module.weight) \
            - self._get_conv_output(low, pos_w) \
            - self._get_conv_output(high, neg_w)

        scale = safe_divide(rel, norm_out)

        rel = inputs * self.gradprop2(inputs, outputs, scale, self.module.weight) \
            - low * self.gradprop2(inputs, outputs, scale, pos_w) \
            - high * self.gradprop2(inputs, outputs, scale, neg_w)

        return rel

    def relprop(self, rel, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        pos_w = torch.clamp(self.module.weight, min=0)
        neg_w = torch.clamp(self.module.weight, max=0)
        pos_x = torch.clamp(inputs, min=0)
        neg_x = torch.clamp(inputs, max=0)

        if inputs.shape[1] == 3:
            return self.final_backward(rel, inputs, outputs, pos_w, neg_w)

        return self.backward(rel, pos_x, neg_x, pos_w, neg_w)


class Repeat(RelPropSimple):
    pass


class GetItem(RelPropSimple):
    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None):
        # If module's output is a tuple, propagate rel as is
        inputs = args[0]
        if not torch.is_tensor(inputs):
            return rel

        rel_fill = torch.zeros_like(inputs)
        rel_fill[args[1]] = rel

        return rel_fill


class Unsqueeze(RelProp):
    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None) -> _TensorOrTensors:
        if 'input' in kwargs:
            del kwargs['input']
        else:
            args = args[1:]

        rel = torch.squeeze(rel, *args, **kwargs)
        return rel


class Expand(RelPropSimple):
    pass


class Permute(RelPropSimple):
    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None) -> _TensorOrTensors:
        dims = kwargs.get('dims', None)
        if dims is None:
            dims = args[1] if isinstance(args[1], (tuple, list)) else args[1:]

        dims = torch.LongTensor(dims)
        inv = torch.empty_like(dims)
        inv[dims] = torch.arange(len(dims), device=dims.device)

        return rel.permute(inv.tolist())


class Reshape(RelProp):
    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None) -> _TensorOrTensors:
        return rel.reshape(inputs.shape)


class GetAttr(RelProp):
    pass


class MultiHeadAttention(RelPropSimple):
    def forward(self):
        pass

    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None) -> _TensorOrTensors:
        rel = super().relprop(rel, inputs, outputs[0], args, kwargs)

        return rel
