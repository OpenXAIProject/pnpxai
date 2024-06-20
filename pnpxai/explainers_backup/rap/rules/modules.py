from collections import OrderedDict
from typing import Sequence, Tuple

from torch import Tensor
import torch
from torch import nn
from torch.nn import functional as F

from pnpxai.explainers_backup.rap.rules.functions import Transpose, Unsqueeze, View, MatMul, SoftMax
from pnpxai.explainers_backup.rap.rules.base import RelProp, RelPropSeparate, RelPropSimple, _TensorOrTensors, safe_divide
from pnpxai.utils import linear_from_params


def parse_arg_by_name(name, idx, args=None, kwargs=None, default=None):
    args = args or []
    kwargs = kwargs or {}
    return kwargs.get(name, args[idx] if len(args) > idx else default)


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
            parse_arg_by_name('need_weights', 4, args, kwargs, True),
            parse_arg_by_name('attn_mask', 5, args, kwargs),
            parse_arg_by_name('average_attn_weights', 6, args, kwargs, True),
            parse_arg_by_name('is_causal', 7, args, kwargs, False),
        )

    def _preprocess(self, rules: list, query: Tensor, key: Tensor, value: Tensor, key_padding_mask=None,  need_weights=True, attn_mask=None, is_causal=False) -> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], list]:
        is_batched = F._mha_shape_check(
            query, key, value, key_padding_mask, attn_mask, self.module.num_heads
        )
        if self.module.batch_first and is_batched:
            rules.append(lambda *rels: [
                Transpose().relprop(rel, x, None, (query, 1, 0), {})
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

    def _qkv_proj(self, rules: list,  query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tuple[Tensor, Tensor, Tensor], list]:
        batch_size = query.shape[1]

        if self.module._qkv_same_embed_dim:
            indices = [(i+1) * self.module.kdim for i in range(3)]
            w_q = self.module.in_proj_weight[:indices[0]]
            w_k = self.module.in_proj_weight[indices[0]:indices[1]]
            w_v = self.module.in_proj_weight[indices[1]:indices[2]]
        else:
            w_q, w_k, w_v = self.module.q_proj_weight, self.module.k_proj_weight, self.module.v_proj_weight

        if self.module.in_proj_bias is not None:
            b_q, b_k, b_v = self.module.in_proj_bias.chunk(3)
        else:
            b_q = b_k = b_v = None

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        rules.append(lambda *rels: [
            Linear(linear_from_params(w_q, b_q))
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

    def _view_with_head(self, rules: list, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tuple[Tensor, Tensor, Tensor], list]:
        batch_size = value.shape[1]
        num_heads = self.module.num_heads
        head_dim = self.module.head_dim

        def _get_view_shape(x: Tensor):
            return x.shape[0], batch_size * num_heads, head_dim

        query_t, key_t, value_t = (
            x.view(_get_view_shape(x)).transpose(0, 1)
            for x in (query, key, value)
        )

        rules.append(lambda *rels: [
            View().relprop(rel, x, None, (x, *_get_view_shape(x)))
            for rel, x in zip(rels, (query, key, value))
        ])
        rules.append(lambda *rels: [
            Transpose().relprop(rel, None, None, (None, 0, 1), {})
            for rel, x in zip(rels, (query, key, value))
        ])

        return (query_t, key_t, value_t), rules

    def _attention(self, rules: list, query: Tensor, key: Tensor, key_padding_mask=None, attn_mask=None) -> Tuple[Tensor, list]:
        batch_size = query.shape[1]
        num_heads = self.module.num_heads
        head_dim = self.module.head_dim

        if self.module.add_zero_attn:
            zero_attn_shape = (batch_size * num_heads, 1, head_dim)
            key_zeros = torch.zeros(
                zero_attn_shape, dtype=key.dtype, device=key.device)
            key = torch.cat([key, key_zeros], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        if key_padding_mask is not None:
            src_len = key.shape(1)
            assert key_padding_mask.shape == (batch_size, src_len), \
                f"expecting key_padding_mask shape of {(batch_size, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len) \
                .expand(-1, num_heads, -1, -1)\
                .reshape(batch_size * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        E = query.shape[-1]
        key_t = key.transpose(-2, -1)
        attn_out = torch.bmm(query, key_t) / (E**.5)

        rules.append(lambda q_rel, k_rel, v_rel: [
            q_rel, Transpose().relprop(k_rel, key, key_t, (key, -2, -1), {}), v_rel
        ])
        rules.append(lambda attn_rel, v_rel: [
            *MatMul().relprop(attn_rel, (query, key_t), attn_out), v_rel
        ])
        if attn_mask is not None:
            attn_out = attn_mask + attn_out

        attn_out_weighted = F.softmax(attn_out, dim=-1)
        rules.append(lambda attn_rel, v_rel: [
            SoftMax().relprop(attn_rel, attn_out, attn_out_weighted, [-1], {}),
            v_rel
        ])

        return attn_out_weighted, rules

    def _attn_v_mul(self, rules: list, attn_weights: Tensor, value: Tensor) -> Tuple[Tensor, list]:
        out = torch.bmm(attn_weights, value)
        rules.append(lambda out_rel: [
            *MatMul().relprop(out_rel, (attn_weights, value), out),
        ])

        return out, rules

    def _out_proj(self, rules: list, out: Tensor, source_shape) -> Tuple[Tensor, list]:
        orig_out = out.clone()
        rules.append(lambda out_rel: [
            View().relprop(out_rel, orig_out, None)
        ])
        rules.append(lambda out_rel: [
            Transpose().relprop(out_rel, None, None, (None, 0, 1), {})
        ])

        out = out.view(source_shape).transpose(0, 1)

        out_lin = self.module.out_proj
        out_proj = out_lin(out)

        rules.append(lambda out_rel: (
            Linear(out_lin).relprop(out_rel, out, out_proj),
        ))

        return out_proj, rules

    def _match_rels_with_inputs(self, rels: Sequence[Tensor], inputs: Sequence[Tensor]):
        rel_map = OrderedDict()
        for rel, datum in zip(rels, inputs):
            idx = id(datum)
            if idx not in rel_map:
                rel_map[idx] = 0
            rel_map[idx] += rel

        return list(rel_map.values())

    def relprop(self, rel: Tensor, inputs: Tensor, outputs: Tensor, args=None, kwargs=None) -> _TensorOrTensors:
        if torch.is_tensor(inputs):
            inputs = (inputs, inputs, inputs)
        query, key, value = inputs
        key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal = self._get_args(
            args, kwargs
        )

        rules = []

        (query, key, value, attn_mask), rules = self._preprocess(
            rules, query, key, value, key_padding_mask, need_weights, attn_mask, is_causal
        )
        source_shape = query.shape

        (query, key, value), rules = self._qkv_proj(rules, query, key, value)

        (query, key, value), rules = self._view_with_head(rules, query, key, value)

        (attn_weights), rules = self._attention(
            rules, query, key, key_padding_mask, attn_mask
        )

        (out), rules = self._attn_v_mul(rules, attn_weights, value)

        (out), rules = self._out_proj(rules, out, source_shape)

        rels = (rel,)
        for rule in reversed(rules):
            rels = rule(*rels)

        rels = self._match_rels_with_inputs(inputs, rels)
        if len(rels) == 1:
            return rels[0]
        
        return rels
