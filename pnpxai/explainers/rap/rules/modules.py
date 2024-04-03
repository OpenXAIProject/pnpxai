from pnpxai.explainers.rap.rules.base import RelProp, RelPropSeparate, RelPropSimple, _TensorOrTensors, safe_divide
from torch import Tensor
import torch.nn.functional as F
import torch


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


class BatchNorm2d(RelProp):
    pass


class LayerNorm(RelProp):
    pass


class Linear(RelPropSeparate):
    def agg_dims(self):
        return -1

    def forward(self, weights: Tensor, inputs: Tensor) -> Tensor:
        return F.linear(weights, inputs)

    def first_prop(self, pd, px, nx, pw, nw):
        Rpp = self.forward(px, pw) * pd
        Rpn = self.forward(px, nw) * pd
        Rnp = self.forward(nx, pw) * pd
        Rnn = self.forward(nx, nw) * pd
        Pos = (Rpp + Rnn).sum(dim=-1, keepdim=True)
        Neg = (Rpn + Rnp).sum(dim=-1, keepdim=True)

        Z1 = self.forward(px, pw)
        Z2 = self.forward(px, nw)
        Z3 = self.forward(nx, pw)
        Z4 = self.forward(nx, nw)

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

    def redistribute(self, Rp_tmp):
        Rp = torch.clamp(Rp_tmp, min=0)
        Rn = torch.clamp(Rp_tmp, max=0)
        R_tot = (Rp - Rn).sum(dim=-1, keepdim=True)
        Rp_tmp3 = safe_divide(Rp, R_tot) * \
            (Rp + Rn).sum(dim=-1, keepdim=True)
        Rn_tmp3 = -safe_divide(Rn, R_tot) * \
            (Rp + Rn).sum(dim=-1, keepdim=True)
        return Rp_tmp3 + Rn_tmp3

    def relprop(self, rel: _TensorOrTensors, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        pos_w = torch.clamp(self.module.weight, min=0)
        neg_w = torch.clamp(self.module.weight, max=0)
        pos_x = torch.clamp(inputs, min=0)
        neg_x = torch.clamp(inputs, max=0)

        if torch.is_tensor(rel) and rel.max() == 1:  # first propagation
            Rp_tmp = self.first_prop(rel, pos_x, neg_x, pos_w, neg_w)
            return self.redistribute(Rp_tmp)

        return self.backward(rel, pos_x, neg_x, pos_w, neg_w)


class Conv2d(RelPropSeparate):
    @property
    def agg_dims(self):
        return list(range(1, 4))

    def forward(self, inputs: Tensor, weights: Tensor) -> Tensor:
        return torch.conv2d(inputs, weights, stride=self.module.stride, padding=self.module.padding)

    def gradprop_trans(self, inputs, outputs, DY, weight):
        padding = self.module.padding
        stride = self.module.stride
        kernel_size = self.module.kernel_size

        output_padding = inputs.size(2) - (
            (outputs.size(2) - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
        )
        return F.conv_transpose2d(DY, weight, stride=stride, padding=padding, output_padding=output_padding)

    def final_backward(self, rel: Tensor, inputs: Tensor, outputs: Tensor, pos_w: Tensor, neg_w: Tensor):
        agg_dims = tuple(range(1, inputs.ndim))
        template = torch.zeros_like(inputs)
        low = template + torch.amin(inputs, dim=agg_dims, keepdim=True)
        high = template + torch.amax(inputs, dim=agg_dims, keepdim=True)

        norm_out = self.forward(inputs, self.module.weight) \
            - self.forward(low, pos_w) \
            - self.forward(high, neg_w)

        scale = safe_divide(rel, norm_out)

        rel = inputs * self.gradprop_trans(inputs, outputs, scale, self.module.weight) \
            - low * self.gradprop_trans(inputs, outputs, scale, pos_w) \
            - high * self.gradprop_trans(inputs, outputs, scale, neg_w)

        return rel

    def relprop(self, rel, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        pos_w = torch.clamp(self.module.weight, min=0)
        neg_w = torch.clamp(self.module.weight, max=0)
        pos_x = torch.clamp(inputs, min=0)
        neg_x = torch.clamp(inputs, max=0)

        if inputs.shape[1] == 3:
            return self.final_backward(rel, inputs, outputs, pos_w, neg_w)

        return self.backward(rel, pos_x, neg_x, pos_w, neg_w)


class MultiHeadAttention(RelPropSimple):
    def forward(self):
        pass

    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None) -> _TensorOrTensors:
        rel = super().relprop(rel, inputs, outputs[0], args, kwargs)

        return rel
