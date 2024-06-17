from abc import abstractmethod
from typing import Callable, Tuple, List, Sequence, Optional, Union

import torch
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn.modules import Module
from zennit.canonizers import Canonizer
from zennit.composites import layer_map_base, LayerMapComposite
from zennit.rules import AlphaBeta
from zennit.types import Linear

from .attentions.attributions import SavingAttentionAttributor
from .attentions.rules import CGWAttentionPropagation
from .attentions.module_converters import default_attention_converters
from .zennit.attribution import Gradient, LayerGradient
from .zennit.base import ZennitExplainer
from .utils import captum_wrap_model_input


class AttentionRolloutBase(ZennitExplainer):
    def __init__(
        self,
        model: Module,
        head_fusion_fn: Optional[Callable[[Tensor], Tensor]]=None,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        n_classes: Optional[int]=None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes,
        )
        self.head_fusion_fn = head_fusion_fn

    @abstractmethod
    def collect_attention_map(self, inputs, targets):
        raise NotImplementedError

    @abstractmethod
    def rollout(self, *args):
        raise NotImplementedError
    
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor
    ) -> Union[Tensor, Tuple[Tensor]]:
        attn_maps = self.collect_attention_map(inputs, targets)
        attrs = self.rollout(*attn_maps)
        return attrs


class AttentionRollout(AttentionRolloutBase):
    def __init__(
        self,
        model: Module,
        head_fusion_fn: Optional[Callable[[Tensor], Tensor]]=None,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        n_classes: Optional[int]=None
    ) -> None:
        head_fusion_fn = head_fusion_fn or self.default_head_fusion_fn
        super().__init__(
            model,
            head_fusion_fn,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes
        )

    @staticmethod
    def default_head_fusion_fn(attns):
        return attns.min(dim=1)[0]
    
    def collect_attention_map(self, inputs, targets):
        # get all attn maps
        with SavingAttentionAttributor(model=self.model) as attributor:
            weights_all = attributor(inputs, None)
        return (weights_all,)
    
    def rollout(self, weights_all):
        sz = weights_all[0].size()
        assert all(
            attn_output_weights.size() == sz
            for attn_output_weights in weights_all
        )
        bsz, num_heads, tgt_len, src_len = sz
        attrs = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
        for attn_output_weights in weights_all:
            fused = self.head_fusion_fn(attn_output_weights)
            identity = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
            attr = .5 * fused + .5 * identity
            attr /= attr.sum(dim=-1, keepdim=True)
            attrs = torch.matmul(attrs, attr)
        return attrs    


class TransformerAttribution(AttentionRolloutBase):
    def __init__(
        self,
        model: Module,
        alpha: float=2.,
        beta: float=1.,
        stabilizer: float=1e-6,
        head_fusion_fn: Optional[Callable[[Tensor], Tensor]]=None,
        zennit_canonizers: Optional[List[Canonizer]]=None,
        layer: Optional[Union[Module, Sequence[Module]]]=None,
        forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        additional_forward_arg_extractor: Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]=None,
        n_classes: Optional[int]=None
    ) -> None:
        head_fusion_fn = head_fusion_fn or self.default_head_fusion_fn
        super().__init__(
            model,
            head_fusion_fn,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes
        )
        self.alpha = alpha
        self.beta = beta
        self.stabilizer = stabilizer
        self.zennit_canonizers = zennit_canonizers or []
        self.layer = layer

    @staticmethod
    def default_head_fusion_fn(attns):
        return attns.mean(dim=1)
    
    @property
    def zennit_composite(self):
        layer_map = [
            (MultiheadAttention, CGWAttentionPropagation(
                alpha=self.alpha,
                beta=self.beta,
                stabilizer=self.stabilizer,
                save_attn_output_weights=False,
            )),
            (Linear, AlphaBeta(
                alpha=self.alpha,
                beta=self.beta,
                stabilizer=self.stabilizer,
            )),
        ] + layer_map_base(stabilizer=self.stabilizer)
        canonizers = default_attention_converters + self.zennit_canonizers
        return LayerMapComposite(layer_map=layer_map, canonizers=canonizers)
    
    @property
    def _layer_gradient(self) -> LayerGradient:
        wrapped_model = captum_wrap_model_input(self.model)
        layers = [
            wrapped_model.input_maps[layer] if isinstance(layer, str)
            else layer for layer in self.layer
        ] if isinstance(self.layer, Sequence) else self.layer
        return LayerGradient(
            model=wrapped_model,
            layer=layers,
            composite=self.zennit_composite,
        )
    
    @property
    def _gradient(self) -> Gradient:
        return Gradient(
            model=self.model,
            composite=self.zennit_composite,
        )

    @property
    def attributor(self):
        if self.layer is None:
            return self._gradient
        return self._layer_gradient
    
    def collect_attention_map(self, inputs, targets):
        forward_args, additional_forward_args = self._extract_forward_args(inputs)
        with self.attributor as attributor:
            attributor.forward(
                forward_args=forward_args,
                targets=targets,
                additional_forward_args=additional_forward_args,
            )
            grads, rels = [], []
            for hook_ref in attributor.composite.hook_refs:
                if isinstance(hook_ref, CGWAttentionPropagation):
                    grads.append(hook_ref.stored_tensors["attn_grads"])
                    rels.append(hook_ref.stored_tensors["attn_rels"])
        return grads, rels

    def rollout(self, grads, rels):
        bsz, num_heads, tgt_len, src_len = grads[0].shape
        assert tgt_len == src_len, "Must be self-attention"
        attrs = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
        for grad, rel in zip(grads, rels):
            grad_x_rel = grad * rel
            fused = self.head_fusion_fn(grad_x_rel)
            identity = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
            attr = .5 * fused + .5 * identity
            attr /= attr.sum(dim=-1, keepdim=True)
            attrs = torch.matmul(attrs, attr)
        return attrs


class GenericAttention(AttentionRolloutBase):
    def __init__(
            self,
            model: Module,
            alpha: float=2.,
            beta: float=1.,
            stabilizer: float=1e-6,
            head_fusion_fn: Optional[Callable[[Tensor], Tensor]]=None,
            n_classes: Optional[int]=None
        ) -> None:
        raise NotImplementedError