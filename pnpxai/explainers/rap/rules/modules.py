from collections import OrderedDict
from typing import Sequence

from torch import Tensor
import torch
from torch import nn
from torch.nn import functional as F

from pnpxai.explainers.rap.rules.functions import Transpose, Unsqueeze
from pnpxai.explainers.rap.rules.base import RelProp, RelPropSeparate, RelPropSimple, _TensorOrTensors, safe_divide
from pnpxai.utils import Struct


def parse_arg_by_name(name, idx, args=None, kwargs=None, default=None):
    args = args or []
    kwargs = kwargs or {}
    return kwargs.get(name, args[idx] if len(args) >= idx + 1 else default)


class Dropout(RelProp):
    pass


class MaxPool2d(RelPropSimple):
    pass


class MaxPool1d(RelPropSimple):
    pass


class AdaptiveAvgPool2d(RelPropSimple):
    pass


class AdaptiveAvgPool1d(RelPropSimple):
    pass


class AvgPool1d(RelPropSimple):
    pass


class AvgPool2d(RelPropSimple):
    pass


class BatchNorm1d(RelProp):
    pass


class BatchNorm2d(RelProp):
    pass


class LayerNorm(RelProp):
    pass


class Linear(RelPropSeparate):
    @property
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
            rel = self.first_prop(rel, pos_x, neg_x, pos_w, neg_w)
            return self.redistribute(rel)

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


class Conv1d(RelPropSeparate):
    @property
    def agg_dims(self):
        return list(range(1, 3))

    def forward(self, inputs: Tensor, weights: Tensor) -> Tensor:
        return torch.conv1d(inputs, weights, stride=self.module.stride, padding=self.module.padding)

    def relprop(self, rel, inputs: _TensorOrTensors, outputs: _TensorOrTensors, args=None, kwargs=None):
        pos_w = torch.clamp(self.module.weight, min=0)
        neg_w = torch.clamp(self.module.weight, max=0)
        pos_x = torch.clamp(inputs, min=0)
        neg_x = torch.clamp(inputs, max=0)

        return self.backward(rel, pos_x, neg_x, pos_w, neg_w)


class MultiHeadAttention(RelPropSimple):
    def __init__(self, module: nn.Module):
        self.module: nn.MultiheadAttention
        super(MultiHeadAttention, self).__init__(module)

    def _get_args(self, args, kwargs):
        return (
            parse_arg_by_name('key_padding_mask', 3, args, kwargs),
            parse_arg_by_name('attn_mask', 4, args, kwargs),
            parse_arg_by_name('need_weights', 5, args, kwargs),
            parse_arg_by_name('average_attn_weights', 6, args, kwargs, True),
            parse_arg_by_name('is_causal', 7, args, kwargs),
        )

    def _preprocess(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask, attn_mask, need_weights, is_causal, rules: list):
        is_batched = F._mha_shape_check(
            query, key, value, key_padding_mask, attn_mask, self.module.num_heads
        )
        if self.module.batch_first and is_batched:
            rules.append(lambda *rels: [
                Transpose().relprop(rel, x, None, (query, 1, 0))
                for rel, x in zip(rels, (query, key, value))
            ])
            query, key, value = (
                x.transpose(1, 0) for x in (query, key, value)
            )

        if not is_batched:
            rules.append(lambda *rels: [
                Unsqueeze().relprop(rel, x, None, (x, 1))
                for rel, x in zip(rels, (query, key, value))
            ])
            query, key, value = (
                x.unsqueeze(1) for x in (query, key, value)
            )
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if is_causal and key_padding_mask is None and not need_weights:
            attn_mask = None
        else:
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

            if key_padding_mask is not None:
                is_causal = False
        return (query, key, value, attn_mask), rules

    def _qkv_proj(self, query: Tensor, key: Tensor, value: Tensor, rules: list):
        batch_size = query.shape[1]

        if self.module._qkv_same_embed_dim:
            indices = [(i+1) * self.module.kdim for i in range(2)]
            w_q = self.module.in_proj_weight[:indices[0]]
            w_k = self.module.in_proj_weight[indices[0]:indices[1]]
            w_v = self.module.in_proj_weight[indices[2]:indices[3]]
        else:
            w_q, w_k, w_v = self.module.q_proj_weight, self.module.k_proj_weight, self.module.v_proj_weight

        if self.module.in_proj_bias is not None:
            b_q, b_k, b_v = self.module.in_proj_bias.chunk(3)
        else:
            b_q = b_k = b_v = None

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        rules.append(
            lambda *rels: [
                Linear(Struct({'weight': w_q, 'bias': b_q}))
                .relprop(rel, x, None)
                for rel, x in zip(rels, (query, key, value))
            ])

        if self.module.bias_k is not None:
            k = torch.cat([k, self.module.bias_k.repeat(1, batch_size, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        return (q, k, v), rules

    def _view_with_head(self, query: Tensor, key: Tensor, value: Tensor, rules: list) -> Tensor:
        data_len, batch_size, _ = value.shape
        num_heads = self.module.num_heads
        head_dim = self.module.head_dim
        return value.view(data_len, batch_size * num_heads, head_dim).transpose(0, 1)

    @staticmethod
    def _attention_weights_forward(module, query, key, bias_k, key_padding_mask, attn_mask):
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        num_heads, head_dim = module.num_heads, module.head_dim

        if module._qkv_same_embed_dim:
            indices = [(i+1)*module.kdim for i in range(2)]
            w_q = module.in_proj_weight[:indices[0]]
            w_k = module.in_proj_weight[indices[0]:indices[1]]
        else:
            w_q, w_k = module.q_proj_weight, module.k_proj_weight

        if module.in_proj_bias is not None:
            b_q, b_k, _ = module.in_proj_bias.chunk(3)
        else:
            b_q = b_k = None

        # in projection
        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)

        # add bias along batch dimension
        if bias_k is not None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None

        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * num_heads, head_dim).transpose(0, 1)

        # add zero attention along batch dimension (now first)
        if module.add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz *
                                                      num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # calculate attention
        B, Nt, E = q.shape
        q_scaled = q / (E**.5)
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        return attn_output_weights

    def _match_rels_with_inputs(self, rels: Sequence[Tensor], inputs: Sequence[Tensor]):
        rel_map = OrderedDict()
        for rel, datum in zip(rels, inputs):
            idx = id(datum)
            if idx not in rel_map:
                rel_map[idx] = 0
            rel_map[idx] += rel

        return list(rel_map.values())

    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None) -> _TensorOrTensors:
        query, key, value = inputs
        key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal = self._get_args(
            args, kwargs
        )

        rules = []

        (query, key, value), rules = self._preprocess(
            query, key, value, key_padding_mask, attn_mask, need_weights, is_causal, rules
        )

        (query, key, value), rules = self._qkv_proj(query, key, value, rules)

        (query, key, value) = self._view_with_head(query, key, value, rules)

        rels = (rel, rel, rel)
        for rule in reversed(rules):
            rels = rule(rels)

        rels = self._match_rels_with_inputs(inputs, rels)

        return rel
