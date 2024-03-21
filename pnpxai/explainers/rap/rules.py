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

    def relprop(self, rel: _TensorOrTensors, inputs: Optional[_TensorOrTensors], outputs: Optional[_TensorOrTensors]) -> _TensorOrTensors:
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

    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors):
        if torch.is_tensor(rel):
            rel = [rel]
        rel = [self.backward(datum, inputs, outputs) for datum in rel]
        if len(rel) == 1:
            return rel[0]

        return rel


class ReLU(RelProp):
    pass


class Dropout(RelProp):
    pass


class MaxPool2d(RelPropSimple):
    pass


class AdaptiveAvgPool2d(RelPropSimple):
    pass


class AvgPool2d(RelPropSimple):
    pass


class Add(RelPropSimple):
    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors):
        inputs = [F.relu(input) for input in inputs]
        outputs = torch.add(*inputs)
        return super().relprop(rel, inputs, outputs)


class Flatten(RelProp):
    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors):
        return rel.reshape(inputs.shape)


class Cat(RelProp):
    def backward(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors):
        Sp = safe_divide(rel, outputs)
        Cp = self.gradprop(outputs, inputs, Sp)
        rel = [x * cp for x, cp in zip(inputs, Cp)]

        return rel


class BatchNorm2d(RelProp):
    def f(self, R, w1, x1):
        Z1 = x1 * w1
        S1 = safe_divide(R, Z1) * w1
        C1 = x1 * S1
        return C1

    def backward(self, rel: Tensor, inputs: _TensorOrTensors, outputs: _TensorOrTensors):
        X = inputs
        weight = self.module.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.module.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2)
             + self.module.eps).pow(0.5))

        if torch.is_tensor(self.module.bias):
            bias = self.module.bias.unsqueeze(-1).unsqueeze(-1)
            bias_p = safe_divide(bias * rel.ne(0).type(self.module.bias.type()),
                                 rel.ne(0).type(self.module.bias.type()).sum(dim=[2, 3], keepdim=True))
            rel = rel - bias_p

        Rp = self.f(rel, weight, X)

        if torch.is_tensor(self.module.bias):
            Bp = self.f(bias_p, weight, X)
            Rp = Rp + Bp

        return Rp


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

    def relprop(self, R_p, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        pw = torch.clamp(self.module.weight, min=0)
        nw = torch.clamp(self.module.weight, max=0)
        X = inputs
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) == True and R_p.max() == 1:  # first propagation
            pd = R_p

            Rp_tmp = self.first_prop(pd, px, nx, pw, nw)
            A = self.redistribute(Rp_tmp)

            return A
        else:
            Rp = self.backward(R_p, px, nx, pw, nw)

        return Rp


class Conv2d(RelProp):
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

    def pos_prop(self, R, Za1, Za2, x1):
        R_pos = torch.clamp(R, min=0)
        R_neg = torch.clamp(R, max=0)
        S1 = safe_divide(
            (R_pos * safe_divide((Za1 + Za2), Za1 + Za2)), Za1)
        C1 = x1 * self.gradprop(Za1, x1, S1)[0]
        S1n = safe_divide(
            (R_neg * safe_divide((Za1 + Za2), Za1 + Za2)), Za2)
        C1n = x1 * self.gradprop(Za2, x1, S1n)[0]
        S2 = safe_divide((R_pos * safe_divide((Za2), Za1 + Za2)), Za2)
        C2 = x1 * self.gradprop(Za2, x1, S2)[0]
        S2n = safe_divide((R_neg * safe_divide((Za2), Za1 + Za2)), Za2)
        C2n = x1 * self.gradprop(Za2, x1, S2n)[0]
        Cp = C1 + C2
        Cn = C2n + C1n
        C = (Cp + Cn)
        C = self.shift_rel(
            C, C.sum(dim=[1, 2, 3], keepdim=True) - R.sum(dim=[1, 2, 3], keepdim=True))
        return C

    def f(self, R, w1, w2, x1, x2):
        R_nonzero = R.ne(0).type(R.type())
        Za1 = F.conv2d(x1, w1, bias=None, stride=self.module.stride,
                       padding=self.module.padding) * R_nonzero
        Za2 = - F.conv2d(x1, w2, bias=None, stride=self.module.stride,
                         padding=self.module.padding) * R_nonzero

        Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.module.stride,
                         padding=self.module.padding) * R_nonzero
        Zb2 = F.conv2d(x2, w2, bias=None, stride=self.module.stride,
                       padding=self.module.padding) * R_nonzero

        C1 = self.pos_prop(R, Za1, Za2, x1)
        C2 = self.pos_prop(R, Zb1, Zb2, x2)
        return C1 + C2

    def backward(self, R_p, px, nx, pw, nw):
        Rp = self.f(R_p, pw, nw, px, nx)
        return Rp

    def final_backward(self, inputs, outputs, R_p, pw, nw, X1):
        X = X1
        L = X * 0 + \
            torch.min(torch.min(torch.min(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                      keepdim=True)[0]
        H = X * 0 + \
            torch.max(torch.max(torch.max(X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                      keepdim=True)[0]
        Za = torch.conv2d(X, self.module.weight, bias=None, stride=self.module.stride, padding=self.module.padding) - \
            torch.conv2d(L, pw, bias=None, stride=self.module.stride, padding=self.module.padding) - \
            torch.conv2d(H, nw, bias=None, stride=self.module.stride,
                         padding=self.module.padding)

        Sp = safe_divide(R_p, Za)

        Rp = X * self.gradprop2(inputs, outputs, Sp, self.module.weight) - L * \
            self.gradprop2(inputs, outputs, Sp, pw) - H * \
            self.gradprop2(inputs, outputs, Sp, nw)
        return Rp

    def relprop(self, R_p, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        pw = torch.clamp(self.module.weight, min=0)
        nw = torch.clamp(self.module.weight, max=0)
        px = torch.clamp(inputs, min=0)
        nx = torch.clamp(inputs, max=0)

        if inputs.shape[1] == 3:
            Rp = self.final_backward(inputs, outputs, R_p, pw, nw, inputs)
        else:
            Rp = self.backward(R_p, px, nx, pw, nw)
        return Rp
