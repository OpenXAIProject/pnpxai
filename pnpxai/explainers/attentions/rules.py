from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from zennit.core import Stabilizer
from zennit.rules import ClampMod, zero_bias, ParamMod, NoMod
from zennit.core import RemovableHandleList, RemovableHandle, Hook

from ..attentions import partial_forwards
from ..zennit.hooks import HookWithKwargs


class SavingAttention(HookWithKwargs):
    def __init__(self, saved_name: str="attn_output_weights"):
        super().__init__()
        self.saved_name = saved_name

    def pre_forward(self, module, input, kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return super().pre_forward(module, input, kwargs)

    def forward(self, module, input, kwargs, output):
        self.stored_tensors[self.saved_name] = output[-1]

    def register(self, module):
        '''Register this instance by registering all hooks to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_pre_hook(self.pre_forward, with_kwargs=True),
            module.register_forward_hook(self.post_forward),
            module.register_forward_hook(self.forward, with_kwargs=True),
        ])


class AttentionRuleBase(HookWithKwargs):
    def __init__(
            self,
            stabilizer=1e-6,
            **kwargs
        ):
        super().__init__()
        self.stabilizer = stabilizer
        self.stabilizer_fn = Stabilizer.ensure(stabilizer)

    @abstractmethod
    def backward(self, module: MultiheadAttention, grad_input, grad_output):
        raise NotImplementedError


class AttentionHeadPropagation(AttentionRuleBase):
    def __init__(
        self,
        input_modifiers=[lambda input: input],
        param_modifiers=[NoMod(zero_params=None)],
        output_modifiers=[lambda output: output],
        gradient_mapper=None,
        reducer=None,
        stabilizer=1e-6
    ):
        super().__init__(stabilizer)
        self.input_modifiers = input_modifiers
        self.param_modifiers = param_modifiers
        self.output_modifiers = output_modifiers
        self.gradient_mapper = gradient_mapper
        self.reducer = reducer

    def backward(self, module, grad_input, grad_output):
        query, key, value = (
            inp.clone().requires_grad_()
            for inp in self.stored_tensors["input"]
        )
        inputs = {}

        with torch.autograd.enable_grad():
            # preprocess
            (
                query, key, value,
                key_padding_mask, attn_mask, is_batched
            ) = partial_forwards.preprocess_inputs(
                module,
                query,
                key,
                value,
                **self.stored_kwargs
            )
            tgt_len, bsz, _ = query.size()
            src_len, _, _ = key.size()

            #
            for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
                input = in_mod(original_input).requires_grad_()
                with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
                    output = modified.forward(input)
                    output = out_mod(output)
                inputs.append(input)
                outputs.append(output)



class ConservativeAttentionPropagation(AttentionRuleBase):
    def __init__(self, stabilizer=1e-6):
        super().__init__(stabilizer)

    def backward(self, module, grad_input, grad_output):
        query, key, value = (
            inp.clone().requires_grad_()
            for inp in self.stored_tensors["input"]
        )
        with torch.autograd.enable_grad():
            preprocessed_inputs = partial_forwards.preprocess_inputs(
                module,
                query,
                key,
                value,
                **self.stored_kwargs
            )
            query, key, value, key_padding_mask, attn_mask, is_batched = preprocessed_inputs
            tgt_len, bsz, _ = query.size()
            src_len, _, _ = key.size()

            # forward
            in_proj_query, in_proj_key, in_proj_value = partial_forwards.in_proj_qkv(
                module, query, key, value,
            )
            attn_output_weights = partial_forwards.attn_output_weights(
                module, in_proj_query, in_proj_key,
                module.bias_k, key_padding_mask, attn_mask,
                tgt_len, bsz, src_len,
            )
            attn_output_weights = attn_output_weights.detach()
            in_proj_output = partial_forwards.in_proj_output(
                module, in_proj_value, attn_output_weights,
                tgt_len, bsz,
            )
            out_proj = partial_forwards.out_proj(
                module, in_proj_output, tgt_len, bsz, is_batched
            )

        # relprop: out_proj -> in_proj_output
        grad_outputs = grad_output[0] / self.stabilizer_fn(out_proj)
        gradients = torch.autograd.grad(
            out_proj,
            in_proj_output,
            grad_outputs=grad_outputs,
            create_graph=grad_output[0].requires_grad,
        )
        rel_in_proj_output = in_proj_output * gradients[0]

        # relprop: in_proj_output -> in_proj_value, attn_output_weights
        grad_outputs = rel_in_proj_output / self.stabilizer_fn(in_proj_output)
        gradients = torch.autograd.grad(
            in_proj_output,
            in_proj_value,
            grad_outputs=grad_outputs,
        )
        rel_in_proj_value = in_proj_value * gradients[0]

        '''
        rel_attn_output_weights = attn_output_weights * gradients[1]

        # relprop: attn_output_weights -> in_proj_query, in_proj_key
        grad_outputs = rel_attn_output_weights / self.stabilizer_fn(attn_output_weights)
        gradients = torch.autograd.grad(
            attn_output_weights,
            (in_proj_query, in_proj_key),
            grad_outputs=grad_outputs,
        )
        rel_in_proj_query = in_proj_query * gradients[0]
        rel_in_proj_key = in_proj_key * gradients[1]

        # relprop: in_proj_query -> query
        grad_outputs = rel_in_proj_query / self.stabilizer_fn(in_proj_query)
        gradients = torch.autograd.grad(
            in_proj_query,
            query,
            grad_outputs=grad_outputs,
        )
        rel_query = query * gradients[0]

        # relprop: in_proj_key -> key
        grad_outputs = rel_in_proj_key / self.stabilizer_fn(in_proj_key)
        gradients = torch.autograd.grad(
            in_proj_key,
            key,
            grad_outputs=grad_outputs,
        )
        rel_key = key * gradients[0]
        '''

        # relprop: in_proj_value -> value
        grad_outputs = rel_in_proj_value / self.stabilizer_fn(in_proj_value)
        gradients = torch.autograd.grad(
            in_proj_value,
            value,
            grad_outputs=grad_outputs,
        )

        rel_value = value * gradients[0]

        if module.batch_first and is_batched:
            query, key, rel_value = (
                x.transpose(1, 0) for x in (query, key, rel_value)
            )
        return (
            torch.zeros_like(query),
            rel_value,
            rel_value,
        )

    def copy(self):
        copy = ConservativeAttentionPropagation.__new__(type(self))
        ConservativeAttentionPropagation.__init__(
            copy,
            self.stabilizer
        )
        return copy


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


class CGWAttentionPropagation(AttentionRuleBase):
    '''
    ``zennit``-compatible attention propagation rule of Chefer, Gur and Wolf.
    '''
    def __init__(
            self,
            alpha=2.,
            beta=1.,
            stabilizer=1e-6,
            save_attn_output_weights=False,
            zero_params=None,
        ):
        super().__init__(stabilizer)
        self.alpha = alpha
        self.beta = beta
        self.stabilizer = stabilizer
        self.save_attn_output_weights = save_attn_output_weights
        self.input_clampers = [
            lambda input: input.clamp(min=0), # px
            lambda input: input.clamp(max=0), # nx
            lambda input: input.clamp(min=0), # px
            lambda input: input.clamp(max=0), # nx
        ]
        self.param_clampers = [
            ClampMod(min=0., zero_params=zero_params), # pw
            ClampMod(max=0., zero_params=zero_bias(zero_params)), # nw
            ClampMod(max=0., zero_params=zero_bias(zero_params)), # nw
            ClampMod(min=0., zero_params=zero_params), # pw
        ]

    def backward(self, module, grad_input, grad_output):
        query, key, value = (
            inp.clone().requires_grad_()
            for inp in self.stored_tensors["input"]
        )
        with torch.autograd.enable_grad():
            preprocessed_inputs = partial_forwards.preprocess_inputs(
                module,
                query,
                key,
                value,
                **self.stored_kwargs
            )
            query, key, value, key_padding_mask, attn_mask, is_batched = preprocessed_inputs
            tgt_len, bsz, _ = query.size()
            src_len, _, _ = key.size()

            # forward
            in_proj_query, in_proj_key, in_proj_value = partial_forwards.in_proj_qkv(
                module, query, key, value,
            )
            attn_output_weights = partial_forwards.attn_output_weights(
                module, in_proj_query, in_proj_key,
                module.bias_k, key_padding_mask, attn_mask,
                tgt_len, bsz, src_len,
            )
            in_proj_output = partial_forwards.in_proj_output(
                module, in_proj_value, attn_output_weights,
                tgt_len, bsz,
            )
            out_proj = partial_forwards.out_proj(
                module, in_proj_output, tgt_len, bsz, is_batched
            )

            # clamped in_proj_qkv
            clamped_qkvs = []
            clamped_in_proj_qkvs = []
            for input_clamper, param_clamper in zip(self.input_clampers, self.param_clampers):
                clamped_qkv = (input_clamper(query), input_clamper(key), input_clamper(value))
                with ParamMod.ensure(param_clamper)(module) as clamped_module:
                    clamped_in_proj_qkv = partial_forwards.in_proj_qkv(clamped_module, *clamped_qkv)
                clamped_qkvs.append(clamped_qkv)
                clamped_in_proj_qkvs.append(clamped_in_proj_qkv)

            # clamped out_proj
            clamped_in_proj_outputs = []
            clamped_out_projs = []
            for input_clamper, param_clamper in zip(self.input_clampers, self.param_clampers):
                clamped_in_proj_output = input_clamper(in_proj_output)
                with ParamMod.ensure(param_clamper)(module.out_proj) as clamped_out_proj_forward:
                    clamped_out_proj = clamped_out_proj_forward(clamped_in_proj_output)
                clamped_in_proj_outputs.append(clamped_in_proj_output)
                clamped_out_projs.append(clamped_out_proj.view(tgt_len, bsz, -1).transpose(0, 1))

        # save attention output weights
        if self.save_attn_output_weights:
            self.stored_tensors["attn_output_weights"] = attn_output_weights.view(bsz, module.num_heads, tgt_len, -1)

        # save attention gradients
        attn_gradients = torch.autograd.grad(
            out_proj,
            attn_output_weights,
            grad_outputs=torch.ones_like(out_proj),
            create_graph=grad_output[0].requires_grad,
            retain_graph=True,
        )
        self.stored_tensors["attn_grads"] = attn_gradients[0].view(bsz, module.num_heads, tgt_len, -1)

        # relevance propagation: out_proj - in_proj_output / alpha-beta rule
        # activator
        act_rel = torch.zeros_like(clamped_in_proj_outputs[0])
        for clamped_in_proj_output, clamped_out_proj in zip(clamped_in_proj_outputs[:2], clamped_out_projs[:2]):
            grad_outputs = grad_output[0] / self.stabilizer_fn(clamped_out_proj)
            gradients = torch.autograd.grad(
                clamped_out_proj,
                clamped_in_proj_output,
                grad_outputs=grad_outputs,
                create_graph=grad_output[0].requires_grad,
            )
            act_rel += clamped_in_proj_output * gradients[0]

        # inhibitor
        inhb_rel = torch.zeros_like(clamped_in_proj_outputs[0])
        for clamped_in_proj_output, clamped_out_proj in zip(clamped_in_proj_outputs[2:], clamped_out_projs[2:]):
            grad_outputs = grad_output[0] / self.stabilizer_fn(clamped_out_proj)
            gradients = torch.autograd.grad(
                clamped_out_proj,
                clamped_in_proj_output,
                grad_outputs=grad_outputs,
                create_graph=grad_output[0].requires_grad,
            )
            inhb_rel += clamped_in_proj_output * gradients[0]
        rel_in_proj_output = self.alpha * act_rel - self.beta * inhb_rel

        # relevance propagation: in_proj_value, attn_output_weights / zero rule
        grad_outputs = rel_in_proj_output / self.stabilizer_fn(in_proj_output)
        gradients = torch.autograd.grad(
            in_proj_output,
            (in_proj_value, attn_output_weights),
            grad_outputs=grad_outputs,
        )
        rel_in_proj_value = in_proj_value * gradients[0]
        rel_attn_output_weights = attn_output_weights * gradients[1]

        rel_in_proj_value /= 2
        rel_attn_output_weights /= 2

        self.stored_tensors["attn_rels"] = rel_attn_output_weights.view(bsz, module.num_heads, tgt_len, -1)

        # relevance propagation: in_proj_query, in_proj_value / zero rule
        grad_outputs = rel_attn_output_weights / self.stabilizer_fn(attn_output_weights)
        gradients = torch.autograd.grad(
            attn_output_weights,
            (in_proj_query, in_proj_key),
            grad_outputs=grad_outputs,
        )
        rel_in_proj_query = in_proj_query * gradients[0]
        rel_in_proj_key = in_proj_key * gradients[1]

        rel_in_proj_query /= 2
        rel_in_proj_key /= 2

        rel_in_proj_qkvs = (rel_in_proj_query, rel_in_proj_key, rel_in_proj_value)

        # relevance propagation: in_proj_qkv / alpha-beta rule
        rel_in_proj_qkv_iterator = zip(
            zip(*clamped_qkvs),
            zip(*clamped_in_proj_qkvs),
            rel_in_proj_qkvs,
        )
        rel_qkvs = []
        for clamped_ins, clamped_outs, rel_in_proj in rel_in_proj_qkv_iterator:
            '''
            The iterator sequentially propagates relevance following alpha-beta rule
                clamped_ins: [p, n, p, n] of {query, key, value}
                clamped_outs: [pp, nn, pn, np] of {query, key, value}
                rel_in_proj: relevance scores of {query, key, value} propagated right before
            '''
            # activator
            act_rel = torch.zeros_like(clamped_ins[0])
            for clamped_in, clamped_out in zip(clamped_ins[:2], clamped_outs[:2]):
                grad_outputs = rel_in_proj / self.stabilizer_fn(clamped_out)
                gradients = torch.autograd.grad(
                    clamped_out,
                    clamped_in,
                    grad_outputs=grad_outputs,
                )
                act_rel += clamped_in * gradients[0]

            # inhibitor
            inhb_rel = torch.zeros_like(clamped_ins[0])
            for clamped_in, clamped_out in zip(clamped_ins[2:], clamped_outs[2:]):
                grad_outputs = rel_in_proj / self.stabilizer_fn(clamped_out)
                gradients = torch.autograd.grad(
                    clamped_out,
                    clamped_in,
                    grad_outputs=grad_outputs,
                )
                inhb_rel += clamped_in * gradients[0]
            rel = self.alpha * act_rel - self.beta * inhb_rel
            rel_qkvs.append(rel)

        if module.batch_first and is_batched:
            rel_qkvs = tuple(rel.transpose(0, 1) for rel in rel_qkvs)
        return rel_qkvs


    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        '''
        copy = CGWAttentionPropagation.__new__(type(self))
        CGWAttentionPropagation.__init__(
            copy,
            self.alpha,
            self.beta,
            self.stabilizer,
            self.save_attn_output_weights,
        )
        return copy