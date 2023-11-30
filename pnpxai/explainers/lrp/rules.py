from typing import Optional, Tuple
import math

import torch
from torch import Tensor
from torch.overrides import handle_torch_function, has_torch_function
import torch.nn.functional as F

from zennit.core import Hook, Stabilizer, RemovableHandleList, RemovableHandle


class HookWithKwargs(Hook):
    '''Base class for hooks to be used to compute layer-wise attributions.'''
    def __init__(self):
        super().__init__()
        self.stored_kwargs = None

    def forward(self, module, input, kwargs, output):
        '''Forward hook to save module in-/outputs.'''
        self.stored_tensors['input'] = input
        self.stored_kwargs = kwargs

    def register(self, module):
        '''Register this instance by registering all hooks to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_pre_hook(self.pre_forward),
            module.register_forward_hook(self.post_forward),
            module.register_forward_hook(self.forward, with_kwargs=True),
        ])
    

class AttentionHeadRule(HookWithKwargs):
    def __init__(
            self,
            stabilizer=1e-6,
        ):
        super().__init__()
        self.stabilizer = stabilizer

    def backward(self, module, grad_input, grad_output):
        query, key, value = (
            inp.clone().requires_grad_()
            for inp in self.stored_tensors["input"]
        )
        with torch.autograd.enable_grad():
            attn_output = self._relevance_conserving_forward(
                module,
                query, key, value,
                **self.stored_kwargs
            )
        stabilizer_fn = Stabilizer.ensure(self.stabilizer)
        grad_outputs = grad_output[0] / stabilizer_fn(attn_output)
        gradients = torch.autograd.grad(
            (attn_output,),
            (value,),
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance = value * gradients[0]
        print(module.__class__, grad_output[0].sum(), relevance.sum())
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)

    def copy(self):
        copy = AttentionHeadRule.__new__(type(self))
        AttentionHeadRule.__init__(
            copy,
            self.stabilizer
        )
        return copy

    # 다음은 out_proj 핸들링한 케이스
    @staticmethod
    def _relevance_conserving_forward(module, query, key, value, **kwargs):
        attn_output, _ = _relevance_conserving_multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=module.embed_dim,
            num_heads=module.num_heads,
            in_proj_weight=module.in_proj_weight if module._qkv_same_embed_dim else None,
            in_proj_bias=module.in_proj_bias if module._qkv_same_embed_dim else None,
            bias_k=module.bias_k,
            bias_v=module.bias_v,
            add_zero_attn=module.add_zero_attn,
            dropout_p=module.dropout,
            out_proj_weight=module.out_proj.weight,
            out_proj_bias=module.out_proj.bias,
            training=module.training,
            key_padding_mask=kwargs.get("key_padding_mask"),
            need_weights=True, # HERE
            attn_mask=kwargs.get("attn_mask"),
            use_separate_proj_weight=not module._qkv_same_embed_dim,
            q_proj_weight=module.q_proj_weight if not module._qkv_same_embed_dim else None,
            k_proj_weight=module.k_proj_weight if not module._qkv_same_embed_dim else None,
            v_proj_weight=module.v_proj_weight if not module._qkv_same_embed_dim else None,
            average_attn_weights=(kwargs.get("average_attn_weights") is True),
            is_causal=(kwargs.get("is_causal") is True)
        )
        return attn_output

    # 다음은 out-proj 핸들링 안한 케이스
    @staticmethod
    def _relevance_conserving_forward(module, query, key, value, **kwargs):
        # import pdb; pdb.set_trace()
        if module._qkv_same_embed_dim:
            indices = [(i+1)*module.kdim for i in range(3)]
            wq = module.in_proj_weight[:indices[0]]
            wk = module.in_proj_weight[indices[0]:indices[1]]
            wv = module.in_proj_weight[indices[1]:indices[2]]
        else:
            wq, wk, wv = module.q_proj_weight, module.k_proj_weight, module.w_proj_weight
        
        # TODO: to matrices operation if possible
        output = []
        for q, k, v in zip(query, key, value):
            q_score = (q @ wq) @ (k @ wk).T / (module.kdim**.5)
            p_score = F.softmax(q_score, dim=0) # [GH] dim? check please
            out = p_score.T.detach() @ v @ wv
            output.append(out)
        return torch.stack(output)

def _relevance_conserving_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            _relevance_conserving_multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)
    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
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

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)


    if attn_mask is not None:
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    if not training:
        dropout_p = 0.0

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)

    assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights.detach(), v) # HERE
    # import pdb; pdb.set_trace()

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    if not is_batched:
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    # import pdb; pdb.set_trace()
    return attn_output, attn_output_weights

class LayerNormRule(Hook):
    def __init__(self, stabilizer=1e-6):
        super().__init__()
        self.stabilizer = stabilizer

    def forward(self, module, input, output):
        '''Forward hook to save module in-/outputs.'''
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        input = self.stored_tensors["input"][0].clone().requires_grad_()
        with torch.autograd.enable_grad():
            output = self._relevance_conserving_forward(module, input)
        stabilizer_fn = Stabilizer.ensure(self.stabilizer)
        grad_outputs = grad_output[0] / stabilizer_fn(output)
        gradients = torch.autograd.grad(
            (output,),
            (input,),
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad
        )
        relevance = input * gradients[0]
        print(module.__class__, grad_output[0].sum(), relevance.sum())
        return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)

    def copy(self):
        copy = LayerNormRule.__new__(type(self))
        LayerNormRule.__init__(
            copy,
            self.stabilizer
        )
        return copy
    
    @staticmethod
    def _relevance_conserving_forward(module, input):
        dims = list(range(1, input.dim()))
        mean = input.mean((-2,-1))[(None,)*len(dims)].permute(*torch.arange(input.ndim-1, -1, -1))
        std = (input.var((-2,-1)) + module.eps).sqrt()[(None,)*len(dims)].permute(*torch.arange(input.ndim-1, -1, -1))
        normed = (input - mean) / std.detach()
        return normed #* module.weight + module.bias


