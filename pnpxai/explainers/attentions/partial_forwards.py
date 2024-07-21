'''
Defines helper functions to forward ``nn.MultiheadAttention`` step by step
in order to backpropagate relevance score.
'''

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention

def preprocess_inputs(
    module: MultiheadAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    **kwargs,
):
    """
    Preprocesses input tensors for multi-head attention computation.

    Args:
        module: The module performing the attention computation.
        query: The query tensor.
        key: The key tensor.
        value: The value tensor.
        **kwargs: Additional keyword arguments for customization.

    Keyword Args:
        key_padding_mask: (Optional) A mask indicating the padding elements in the key tensor.
        need_weights: (Optional) A flag indicating whether weights are needed.
        attn_mask: (Optional) An attention mask to be applied.
        is_causal: (Optional) A flag indicating whether the attention computation is causal.

    Returns:
        tuple: A tuple containing:
            - query: The preprocessed query tensor.
            - key: The preprocessed key tensor.
            - value: The preprocessed value tensor.
            - key_padding_mask: The preprocessed key padding mask tensor.
            - attn_mask: The preprocessed attention mask tensor.
    """
    
    # parse kwargs
    key_padding_mask = kwargs.get('key_padding_mask')
    need_weights = kwargs.get('need_weights')
    attn_mask = kwargs.get('attn_mask')
    is_causal = kwargs.get('is_causal')

    # Preprocess
    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, module.num_heads)
    if module.batch_first and is_batched:
        query, key, value = (x.transpose(1, 0) for x in (query, key, value))
    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
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
    return query, key, value, key_padding_mask, attn_mask, is_batched


def in_proj_qkv(module: MultiheadAttention, query: Tensor, key: Tensor, value: Tensor):
    """
    Projects query, key, and value tensors for multi-head attention computation.

    Args:
        module: The module performing the multi-head attention computation.
        query: The query tensor preprocessed by ``preprocess_inputs``.
        key: The key tensor preprocessed by ``preprocess_inputs``.
        value: The value tensor preprocessed by ``preprocess_inputs``.

    Returns:
        tuple: A tuple containing:
            - in_proj_query: The projected query tensor.
            - in_proj_key: The projected key tensor.
            - in_proj_value: The projected value tensor.
    """
    if module._qkv_same_embed_dim:
        w_q, w_k, w_v = module.in_proj_weight.chunk(3)
    else:
        w_q, w_k, w_v = module.q_proj_weight, module.k_proj_weight, module.v_proj_weight

    if module.in_proj_bias is not None:
        b_q, b_k, b_v = module.in_proj_bias.chunk(3)
    else:
        b_q = b_k = b_v = None

    # in projection
    return F._in_projection(query, key, value, w_q, w_k, w_v, b_q, b_k, b_v)


def attn_output_weights(
        module,
        in_proj_query,
        in_proj_key,
        bias_k,
        key_padding_mask,
        attn_mask,
        tgt_len,
        bsz,
        src_len,
    ):
    """
    Calculates attention output weights for multi-head attention.

    Args:
        module: The module performing the multi-head attention computation.
        in_proj_query: The projected query tensor from ``in_proj_qkv``.
        in_proj_key: The projected key tensor from ``in_proj_qkv``.
        bias_k: The bias tensor for key.
        key_padding_mask: A mask indicating padding elements in the key tensor preprocessed by ``preprocess_inputs``.
        attn_mask: An attention mask to be applied preprocessed by ``preprocess_inputs``.
        tgt_len: The length of the target sequence.
        bsz: Batch size.
        src_len: The length of the source sequence.

    Returns:
        Tensor: The attention output weights tensor.
    """
    # add bias along batch dimension
    if bias_k is not None:
        in_proj_key = torch.cat([in_proj_key, bias_k.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
    in_proj_query = in_proj_query.view(
        tgt_len,
        bsz * module.num_heads,
        module.head_dim
    ).transpose(0, 1)
    in_proj_key = in_proj_key.view(
        src_len,
        bsz * module.num_heads,
        module.head_dim
    ).transpose(0, 1)

    # add zero attention along batch dimension (now first)
    if module.add_zero_attn:
        zero_attn_shape = (bsz * module.num_heads, 1, module.head_dim)
        in_proj_key = torch.cat([
            in_proj_key,
            torch.zeros(
                zero_attn_shape,
                dtype=in_proj_key.dtype,
                device=in_proj_key.device
            )],
            dim=1
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = in_proj_key.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, module.num_heads, -1, -1).reshape(bsz * module.num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # calculate attention
    B, Nt, E = in_proj_query.shape
    q_scaled = in_proj_query / (E**.5)
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(
            attn_mask, q_scaled, in_proj_key.transpose(-2, -1)
        )
    else:
        attn_output_weights = torch.bmm(q_scaled, in_proj_key.transpose(-2, -1))
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    if module.dropout > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=module.dropout)
    return attn_output_weights


def in_proj_output(
        module,
        in_proj_value,
        attn_output_weights,
        tgt_len,
        bsz,
    ):    
    """
    Projects the output of the attention computation.

    Args:
        module: The module performing the multi-head attention computation.
        in_proj_value: The projected value tensor from ``in_proj_qkv``.
        attn_output_weights: The attention output weights tensor calculated by ``attn_output_weights``.
        tgt_len: The length of the target sequence.
        bsz: Batch size.

    Returns:
        Tensor: The projected output tensor.
    """
    in_proj_value = in_proj_value.view(
        in_proj_value.shape[0],
        bsz * module.num_heads,
        module.head_dim
    ).transpose(0, 1)
    if module.add_zero_attn:
        zero_attn_shape = (bsz * module.num_heads, 1, module.head_dim)
        in_proj_value = torch.cat([
            in_proj_value,
            torch.zeros(
                zero_attn_shape,
                dtype=in_proj_value.dtype,
                device=in_proj_value.device
            )],
            dim=1
        )
    in_proj_output = torch.bmm(attn_output_weights, in_proj_value)
    in_proj_output = in_proj_output.transpose(0, 1).contiguous().view(tgt_len * bsz, -1)
    return in_proj_output


def out_proj(module, in_proj_output, tgt_len, bsz, is_batched):
    """
    Projects the output of the attention computation to the final output tensor.

    Args:
        module: The module performing the multi-head attention computation.
        in_proj_output: The projected output tensor from ``in_proj_output``.
        tgt_len: The length of the target sequence.
        bsz: Batch size.

    Returns:
        Tensor: The final output tensor after projection.
    """
    attn_output = F.linear(
        in_proj_output,
        module.out_proj.weight,
        module.out_proj.bias
    )
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    if module.batch_first and is_batched:
        return attn_output.transpose(0, 1)
    return attn_output
