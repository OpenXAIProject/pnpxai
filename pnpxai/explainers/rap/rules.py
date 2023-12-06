import torch
import torch.nn as nn
from torch import nn, Tensor
from typing import Sequence, Union, Optional
import torch.nn.functional as F


__all__ = ['Clone', 'Cat', 'ReLU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide']

_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]


def safe_divide(a: Tensor, b: Tensor):
    return a / (b + b.eq(0).type(b.type()) * 1e-9) * b.ne(0).type(b.type())



class RelProp:
    captures_in_out: bool = True

    def __init__(self, module: Optional[nn.Module] = None):
        super(RelProp, self).__init__()
        # if not self.training:
        self.X: _TensorOrTensors
        self.Y: _TensorOrTensors
        self.module = module

    def forward_hook(self, module: nn.Module, inputs: _TensorOrTensors, outputs: _TensorOrTensors):
        self.X_orig = inputs[0]

        if type(inputs[0]) in (list, tuple):
            self.X = []
            for i in inputs[0]:
                x = i.detach()
                x.requires_grad = True
                self.X.append(x)
        else:
            self.X = inputs[0].detach()
            self.X.requires_grad = True

        self.Y = outputs

    def gradprop(self, Z: _TensorOrTensors, X: _TensorOrTensors, S: _TensorOrTensors) -> _TensorOrTensors:
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R_p: _TensorOrTensors, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None) -> _TensorOrTensors:
        return R_p


class RelPropSimple(RelProp):
    def relprop(self, r: _TensorOrTensors, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        inputs = inputs if inputs is not None else self.X

        def backward(_r):
            if self.module is None and (inputs is None or outputs is None):
                return r

            Z = outputs if outputs is not None else self.module(inputs)
            Sp = safe_divide(_r, Z)

            Cp = self.gradprop(Z, inputs, Sp)[0]
            if torch.is_tensor(inputs) == False:
                Rp = []
                Rp.append(inputs[0] * Cp)
                Rp.append(inputs[1] * Cp)
            else:
                Rp = inputs * (Cp)
            return Rp
        if torch.is_tensor(r) == False:
            idx = len(r)
            tmp_r = r
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_r[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(r)
        return Rp


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
    def __init__(self, module: Optional[nn.Module] = None):
        module = module or (lambda x: torch.add(*x))
        super().__init__(module)


class Flatten(RelProp):
    captures_in_out: bool = True

    def __init__(self, module: Optional[nn.Module] = None):
        module = module or torch.flatten
        super().__init__(module)

    def relprop(self, r: _TensorOrTensors, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        return r.reshape(inputs.shape)


class Cat(RelProp):
    captures_in_out: bool = False

    def relprop(self, R_p, inputs: Tensor, outputs: Tensor):
        def backward(R_p):
            Z = outputs
            Sp = safe_divide(R_p, Z)

            Cp = self.gradprop(Z, inputs, Sp)

            Rp = []

            for x, cp in zip(self.X, Cp):
                Rp.append(x * (cp))

            return Rp
        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class Sequential(RelProp):
    def relprop(self, r: _TensorOrTensors, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        if self.module is None:
            return r

        for m in reversed(self.module._modules.values()):
            if hasattr(m, 'relprop'):
                r = m.relprop(r)
            elif hasattr(m, 'rule') and hasattr(m.rule, 'relprop'):
                r = m.rule.relprop(r)

        return r


class BatchNorm2d(RelProp):
    def relprop(self, R_p: Tensor, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        def f(R, w1, x1):
            Z1 = x1 * w1
            S1 = safe_divide(R, Z1) * w1
            C1 = x1 * S1
            return C1

        def backward(R_p: Tensor):
            X = self.X

            weight = self.module.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.module.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.module.eps).pow(0.5))

            if torch.is_tensor(self.module.bias):
                bias = self.module.bias.unsqueeze(-1).unsqueeze(-1)
                bias_p = safe_divide(bias * R_p.ne(0).type(self.module.bias.type()),
                                     R_p.ne(0).type(self.module.bias.type()).sum(dim=[2, 3], keepdim=True))
                R_p = R_p - bias_p

            Rp = f(R_p, weight, X)

            if torch.is_tensor(self.module.bias):
                Bp = f(bias_p, weight, X)

                Rp = Rp + Bp

            return Rp

        if torch.is_tensor(R_p) == False:
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class Linear(RelProp):
    def relprop(self, R_p, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(
                R_nonzero, dim=-1, keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K

        def pos_prop(R: Tensor, Za1: Tensor, Za2: Tensor, x1: Tensor):
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
            C = shift_rel(C, C.sum(dim=-1, keepdim=True) -
                          R.sum(dim=-1, keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.linear(x1, w1) * R_nonzero
            Za2 = - F.linear(x1, w2) * R_nonzero

            Zb1 = - F.linear(x2, w1) * R_nonzero
            Zb2 = F.linear(x2, w2) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)

            return C1 + C2

        def first_prop(pd, px, nx, pw, nw):
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

        def backward(R_p, px, nx, pw, nw):
            # dealing bias
            # if torch.is_tensor(self.bias):
            #     bias_p = self.bias * R_p.ne(0).type(self.bias.type())
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp

        def redistribute(Rp_tmp):
            Rp = torch.clamp(Rp_tmp, min=0)
            Rn = torch.clamp(Rp_tmp, max=0)
            R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
            Rp_tmp3 = safe_divide(Rp, R_tot) * \
                (Rp + Rn).sum(dim=-1, keepdim=True)
            Rn_tmp3 = -safe_divide(Rn, R_tot) * \
                (Rp + Rn).sum(dim=-1, keepdim=True)
            return Rp_tmp3 + Rn_tmp3
        pw = torch.clamp(self.module.weight, min=0)
        nw = torch.clamp(self.module.weight, max=0)
        X = self.X
        px = torch.clamp(X, min=0)
        nx = torch.clamp(X, max=0)
        if torch.is_tensor(R_p) == True and R_p.max() == 1:  # first propagation
            pd = R_p

            Rp_tmp = first_prop(pd, px, nx, pw, nw)
            A = redistribute(Rp_tmp)

            return A
        else:
            Rp = backward(R_p, px, nx, pw, nw)

        return Rp


class Conv2d(RelProp):
    def gradprop2(self, DY, weight):
        Z = self.module.forward(self.X)

        output_padding = self.X.size()[2] - (
            (Z.size()[2] - 1) * self.module.stride[0] - 2 * self.module.padding[0] + self.module.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.module.stride, padding=self.module.padding, output_padding=output_padding)

    def relprop(self, R_p, inputs: Optional[_TensorOrTensors] = None, outputs: Optional[_TensorOrTensors] = None):
        def shift_rel(R, R_val):
            R_nonzero = torch.ne(R, 0).type(R.type())
            shift = safe_divide(R_val, torch.sum(
                R_nonzero, dim=[1, 2, 3], keepdim=True)) * torch.ne(R, 0).type(R.type())
            K = R - shift
            return K

        def pos_prop(R, Za1, Za2, x1):
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
            C = shift_rel(
                C, C.sum(dim=[1, 2, 3], keepdim=True) - R.sum(dim=[1, 2, 3], keepdim=True))
            return C

        def f(R, w1, w2, x1, x2):
            R_nonzero = R.ne(0).type(R.type())
            Za1 = F.conv2d(x1, w1, bias=None, stride=self.module.stride,
                           padding=self.module.padding) * R_nonzero
            Za2 = - F.conv2d(x1, w2, bias=None, stride=self.module.stride,
                             padding=self.module.padding) * R_nonzero

            Zb1 = - F.conv2d(x2, w1, bias=None, stride=self.module.stride,
                             padding=self.module.padding) * R_nonzero
            Zb2 = F.conv2d(x2, w2, bias=None, stride=self.module.stride,
                           padding=self.module.padding) * R_nonzero

            C1 = pos_prop(R, Za1, Za2, x1)
            C2 = pos_prop(R, Zb1, Zb2, x2)
            return C1 + C2

        def backward(R_p, px, nx, pw, nw):

            # if torch.is_tensor(self.bias):
            #     bias = self.bias.unsqueeze(-1).unsqueeze(-1)
            #     bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
            #                          R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
            #     R_p = R_p - bias_p

            Rp = f(R_p, pw, nw, px, nx)

            # if torch.is_tensor(self.bias):
            #     Bp = f(bias_p, pw, nw, px, nx)
            #
            #     Rp = Rp + Bp
            return Rp

        def final_backward(R_p, pw, nw, X1):
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

            Rp = X * self.gradprop2(Sp, self.module.weight) - L * \
                self.gradprop2(Sp, pw) - H * self.gradprop2(Sp, nw)
            return Rp
        pw = torch.clamp(self.module.weight, min=0)
        nw = torch.clamp(self.module.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        if self.X.shape[1] == 3:
            Rp = final_backward(R_p, pw, nw, self.X)
        else:
            Rp = backward(R_p, px, nx, pw, nw)
        return Rp
