import torch
import torch.nn.functional as F
from zennit.core import Hook, Stabilizer

from pnpxai.explainers.zennit.hooks import HookWithKwargs


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
        key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal = self._parse_kwargs(self.stored_kwargs)

        with torch.autograd.enable_grad():
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
            # Forward
            attn_output_weights = self._attention_weights_forward(module, query, key, module.bias_k, key_padding_mask, attn_mask)

            in_proj_intermediate = self._in_proj_intermediate_forward(module, value, attn_output_weights)
            in_proj_output = self._in_proj_output_forward(module, in_proj_intermediate, attn_output_weights)

            attn_output = self._out_projection_forward(module, in_proj_output, query.shape[0], query.shape[1])

        stabilizer_fn = Stabilizer.ensure(self.stabilizer)

        # Relevance: in_proj_output
        grad_outputs = grad_output[0] / stabilizer_fn(attn_output)
        gradients = torch.autograd.grad(
            attn_output,
            in_proj_output,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance_in_proj_output = in_proj_output * gradients[0]

        # Relevance: in_proj_intermediate
        grad_outputs = relevance_in_proj_output[0] / stabilizer_fn(in_proj_output)
        gradients = torch.autograd.grad(
            in_proj_output,
            in_proj_intermediate,
            grad_outputs=grad_outputs,
        )
        relevance_in_proj_intermediate = in_proj_intermediate * gradients[0]

        # Relevance: value  
        grad_outputs = relevance_in_proj_intermediate[0] / stabilizer_fn(in_proj_intermediate)
        gradients = torch.autograd.grad(
            in_proj_intermediate,
            value,
            grad_outputs=grad_outputs,
        )
        relevance_value = value * gradients[0]

        if module.batch_first and is_batched:
            query, key, relevance_value = (x.transpose(1, 0) for x in (query, key, relevance_value))

        return (
            torch.zeros_like(query),
            torch.zeros_like(key),
            relevance_value,
        )

    def _parse_kwargs(self, kwargs):
        key_padding_mask = kwargs.get('key_padding_mask')
        need_weights = kwargs.get('need_weights')
        attn_mask = kwargs.get('attn_mask')
        average_attn_weights = kwargs.get('average_attn_weights')
        is_causal = kwargs.get('is_causal')
        return key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal

    def copy(self):
        copy = AttentionHeadRule.__new__(type(self))
        AttentionHeadRule.__init__(
            copy,
            self.stabilizer
        )
        return copy

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
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
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
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # calculate attention
        B, Nt, E = q.shape
        q_scaled = q / (E**.5)
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        return attn_output_weights

    @staticmethod
    def _in_proj_intermediate_forward(module, value, attn_output_weights):
        src_len, bsz, embed_dim = value.shape
        num_heads, head_dim = module.num_heads, module.head_dim
        tgt_len = attn_output_weights.shape[1]

        # (bsz * num_heads, tgt_len, src_len) > (bsz, num_heads * tgt_len, src_len)
        attn_output_weights = attn_output_weights.view(bsz, num_heads * tgt_len, src_len)
        
        # (src_len, bsz, embed_dim) > (bsz, src_len, embed_dim)
        value = value.transpose(0, 1)

        # (bsz, num_heads * tgt_len, embed_dim)
        in_proj_intermediate = torch.bmm(attn_output_weights, value)

        # (bsz, num_heads, tgt_len, embed_dim)
        in_proj_intermediate = in_proj_intermediate.view(bsz, num_heads, tgt_len, embed_dim)

        # (bsz * num_heads, tgt_len, embed_dim)
        in_proj_intermediate = in_proj_intermediate.view(bsz * num_heads, tgt_len, embed_dim)

        return in_proj_intermediate

    @staticmethod
    def _in_proj_output_forward(module, in_proj_intermediate, attn_output_weights):
        if module._qkv_same_embed_dim:
            w_v = module.in_proj_weight[2*module.kdim:3*module.kdim]
        else:
            w_v = module.v_proj_weight

        if module.in_proj_bias is not None:
            _, _, b_v = module.in_proj_bias.chunk(3)
        else:
            b_v = None
        num_heads, head_dim = module.num_heads, module.head_dim
        embed_dim = w_v.shape[0]
        bsz_num_heads, tgt_len, src_len = attn_output_weights.shape
        bsz = bsz_num_heads // num_heads

        # (num_heads, bsz * tgt_len, embed_dim)
        in_proj_intermediate = in_proj_intermediate.view(bsz, num_heads, tgt_len, embed_dim).transpose(0, 1).contiguous().view(num_heads, bsz * tgt_len, embed_dim)
        
        # (num_heads, embed_dim, head_dim)
        w_v = w_v.T.view(embed_dim, num_heads, head_dim).transpose(0, 1)
        
        # (num_heads, bsz * tgt_len, head_dim)
        in_proj_output = torch.bmm(in_proj_intermediate, w_v)
        
        # (tgt_len, bsz, num_heads, head_dim)
        in_proj_output = in_proj_output.view(num_heads, bsz, tgt_len, head_dim).permute(2, 1, 0, 3)

        if module.in_proj_bias is not None:
            # (bsz * num_heads, src_len, head_dim)
            b_v = torch.tile(b_v, (src_len, bsz, 1)).view(src_len, bsz * num_heads, head_dim).transpose(0, 1)

            # (tgt_len, bsz, num_heads, head_dim)
            scaled_b_v = torch.bmm(attn_output_weights, b_v).view(bsz, num_heads, tgt_len, head_dim).permute(2, 0, 1, 3)

            # (tgt_len, bsz, num_heads, head_dim)
            in_proj_output += scaled_b_v

        # (tgt_len * bsz, embed_dim)
        in_proj_output = in_proj_output.contiguous().view(tgt_len * bsz, embed_dim)

        return in_proj_output

    @staticmethod
    def _out_projection_forward(module, in_proj_output, tgt_len, bsz):
        attn_output = F.linear(in_proj_output, module.out_proj.weight, module.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1)).transpose(0, 1)
        return attn_output


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
            # Preprocess
            dims = tuple(range(-len(module.normalized_shape), -2, -1))
            
            # Normalization: centering
            mean = input.mean(dims, keepdims=True)
            centered = input - mean

            # Normalization: rescaling
            # correction should be 1 as it is calculated via the unbiased estimator.
            var = input.var(dims, keepdims=True, correction=1).detach()
            rescaled = centered / (var + module.eps).sqrt()

            # Elementwise affine transformation
            transformed = torch.mul(rescaled, module.weight) + module.bias

        stabilizer_fn = Stabilizer.ensure(self.stabilizer)

        ''' # Separate relevance propagation including affine transformation
        # Relevance: rescaled
        grad_outputs = grad_output[0] / stabilizer_fn(transformed)
        gradients = torch.autograd.grad(
            transformed,
            rescaled,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance_rescaled = rescaled * gradients[0]

        # Relevance: input
        # pdb.set_trace()
        grad_outputs = relevance_rescaled[0] / stabilizer_fn(rescaled)
        gradients = torch.autograd.grad(
            rescaled,
            input,
            grad_outputs=grad_outputs,
        )
        relevance_input = input * gradients[0]
        '''

        # ''' # End-to-end relevance propagation including affine transformation
        # Relevance: end-to-end
        grad_outputs = grad_output[0] / stabilizer_fn(transformed)
        gradients = torch.autograd.grad(
            transformed,
            input,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance_input = input * gradients[0]
        # '''

        ''' # Relevance: end-to-end without affine transformation
        grad_outputs = grad_output[0] / stabilizer_fn(rescaled)
        gradients = torch.autograd.grad(
            rescaled,
            input,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        relevance_input = input * gradients[0]
        '''
        return tuple(relevance_input if original.shape == relevance_input.shape else None for original in grad_input)