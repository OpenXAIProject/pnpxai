from abc import abstractmethod
from typing import Callable, Tuple, List, Sequence, Optional, Union, Literal

import torch
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn.modules import Module
from captum.attr import LayerAttribution
from zennit.canonizers import Canonizer
from zennit.composites import layer_map_base, LayerMapComposite
from zennit.rules import AlphaBeta
from zennit.types import Linear
from optuna.trial import Trial

from pnpxai.core.detector.types import Attention
from pnpxai.explainers.attentions.attributions import SavingAttentionAttributor
from pnpxai.explainers.attentions.rules import CGWAttentionPropagation
from pnpxai.explainers.attentions.module_converters import default_attention_converters
from pnpxai.explainers.zennit.attribution import Gradient, LayerGradient
from pnpxai.explainers.zennit.base import ZennitExplainer
from pnpxai.explainers.utils import captum_wrap_model_input
from pnpxai.explainers.types import ForwardArgumentExtractor

def rollout_min_head_fusion_function(attn_weights):
    return attn_weights.min(axis=1).values

def rollout_max_head_fusion_function(attn_weights):
    return attn_weights.max(axis=1).values

def rollout_mean_head_fusion_function(attn_weights):
    return attn_weights.mean(axis=1)

def _get_rollout_head_fusion_function(method: Literal['min', 'max', 'mean']):
    if method == 'min':
        return rollout_min_head_fusion_function
    elif method == 'max':
        return rollout_max_head_fusion_function
    elif method == 'mean':
        return rollout_mean_head_fusion_function


class AttentionRolloutBase(ZennitExplainer):
    """
    Base class for `AttentionRollout` and `TransformerAttribution` explainers.

    Supported Modules: `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        interpolate_mode (Optional[str]): The interpolation mode used by the explainer. Available methods are: "bilinear" and "bicubic"
        head_fusion_method: (Optional[str]): Method to apply to head fusion. Available methods are: `"min"`, `"max"`, `"mean"`
        discard_ratio: (Optional[float]): Describes ration of attention values to discard.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
        n_classes: (Optional[int]): Number of classes
        forward_arg_extractor: A function that extracts forward arguments from the input batch(s) where the attribution scores are assigned.
        additional_forward_arg_extractor: A secondary function that extract additional forward arguments from the input batch(s).
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Samira Abnar, Willem Zuidema. Quantifying Attention Flow in Transformers.
    """

    SUPPORTED_MODULES = [Attention]

    def __init__(
        self,
        model: Module,
        interpolate_mode: Literal['bilinear']='bilinear',
        head_fusion_method: Literal['min', 'max', 'mean']='min',
        discard_ratio: float=0.9,
        forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        n_classes: Optional[int]=None,
    ) -> None:
        super().__init__(
            model,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes,
        )
        self.interpolate_mode = interpolate_mode
        self.head_fusion_method = head_fusion_method
        self.discard_ratio = discard_ratio

    @property
    def head_fusion_function(self):
        return _get_rollout_head_fusion_function(self.head_fusion_method)

    @abstractmethod
    def collect_attention_map(self, inputs, targets):
        raise NotImplementedError

    @abstractmethod
    def rollout(self, *args):
        raise NotImplementedError

    def _discard(self, fused_attn_map):
        org_size = fused_attn_map.size() # keep size to recover it after discard
        flattened = fused_attn_map.flatten(1)
        bsz, n_tokens = flattened.size()
        attn_cls = flattened[:, 0] # keep attn scores of cls token to recover them after discard
        _, indices = flattened.topk(
            k=int(n_tokens*self.discard_ratio),
            dim=-1,
            largest=False,
        )
        flattened[torch.arange(bsz)[:, None], indices] = 0. # discard
        flattened[:, 0] = attn_cls # recover attn scores of cls token
        discarded = flattened.view(*org_size)
        return discarded
    
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor]],
        targets: Tensor
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        Computes attributions for the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels for the inputs.

        Returns:
            torch.Tensor: The result of the explanation.
        """

        attn_maps = self.collect_attention_map(inputs, targets)
        with torch.no_grad():
            rollout = self.rollout(*attn_maps)

        # attn btw cls and patches
        attrs = rollout[:, 0, 1:]
        n_patches = attrs.size(-1)        
        bsz, _, h, w = inputs.size()
        p_h = int(h / w * n_patches ** .5)
        p_w = n_patches // p_h
        attrs = attrs.view(bsz, 1, p_h, p_w)

        # upsampling
        attrs = LayerAttribution.interpolate(
            layer_attribution=attrs,
            interpolate_dims=(h, w),
            interpolate_mode=self.interpolate_mode,
        )
        return attrs

    def get_tunables(self):
        """
        Provides Tunable parameters for the optimizer

        Tunable parameters:
            `interpolate_mode` (str): Value can be selected of `"bilinear"` and `"bicubic"`

            `head_fusion_method` (str): Value can be selected of `"min"`, `"max"`, and `"mean"`

            `discard_ratio` (float): Value can be selected in the range of `range(0, 0.95, 0.05)`
        """
        return {
            'interpolate_mode': (list, {'choices': ['bilinear', 'bicubic']}),
            'head_fusion_method': (list, {'choices': ['min', 'max', 'mean']}),
            'discard_ratio': (float, {'low': 0., 'high': .95, 'step': .05}),
        }


class AttentionRollout(AttentionRolloutBase):
    """
    Implementation of `AttentionRollout` explainer.

    Supported Modules: `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        interpolate_mode (Optional[str]): The interpolation mode used by the explainer. Available methods are: "bilinear" and "bicubic"
        head_fusion_method: (Optional[str]): Method to apply to head fusion. Available methods are: `"min"`, `"max"`, `"mean"`
        discard_ratio: (Optional[float]): Describes ration of attention values to discard.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
        n_classes: (Optional[int]): Number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Samira Abnar, Willem Zuidema. Quantifying Attention Flow in Transformers.
    """
    def __init__(
        self,
        model: Module,
        interpolate_mode: Literal['bilinear']='bilinear',
        head_fusion_method: Literal['min', 'max', 'mean']='min',
        discard_ratio: float=0.9,
        forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        n_classes: Optional[int]=None,
    ) -> None:
        super().__init__(
            model,
            interpolate_mode,
            head_fusion_method,
            discard_ratio,
            forward_arg_extractor,
            additional_forward_arg_extractor,
            n_classes
        )

    def collect_attention_map(self, inputs, targets):
        # get all attn maps
        with SavingAttentionAttributor(model=self.model) as attributor:
            weights_all = attributor(inputs, None)
        return (weights_all,)
    
    def rollout(self, weights_all):
        sz = weights_all[0].size()
        assert all(
            attn_weights.size() == sz
            for attn_weights in weights_all
        )
        bsz, num_heads, tgt_len, src_len = sz
        rollout = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
        for attn_weights in weights_all:
            attn_map = self.head_fusion_function(attn_weights)
            attn_map = self._discard(attn_map)
            identity = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
            attn_map = .5 * attn_map + .5 * identity
            attn_map /= attn_map.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(rollout, attn_map)
        return rollout


class TransformerAttribution(AttentionRolloutBase):
    """
    Implementation of `TransformerAttribution` explainer.

    Supported Modules: `Attention`

    Parameters:
        model (Module): The PyTorch model for which attribution is to be computed.
        interpolate_mode (Optional[str]): The interpolation mode used by the explainer. Available methods are: "bilinear" and "bicubic"
        head_fusion_method: (Optional[str]): Method to apply to head fusion. Available methods are: `"min"`, `"max"`, `"mean"`
        discard_ratio: (Optional[float]): Describes ration of attention values to discard.
        forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract forward arguments from inputs.
        additional_forward_arg_extractor (Optional[Callable[[Tuple[Tensor]], Union[Tensor, Tuple[Tensor]]]]): Optional function to extract additional forward arguments.
        n_classes: (Optional[int]): Number of classes
        **kwargs: Keyword arguments that are forwarded to the base implementation of the Explainer

    Reference:
        Chefer H., Gur S., and Wolf L. Self-Attention Attribution: Transformer interpretability beyond attention visualization.
    """

    SUPPORTED_MODULES = [Attention]
    
    def __init__(
        self,
        model: Module,
        interpolate_mode: Literal['bilinear']='bilinear',
        head_fusion_method: Literal['min', 'max', 'mean']='mean',
        discard_ratio: float=0.9,
        alpha: float=2.,
        beta: float=1.,
        stabilizer: float=1e-6,
        zennit_canonizers: Optional[List[Canonizer]]=None,
        layer: Optional[Union[Module, Sequence[Module]]]=None,
        forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        additional_forward_arg_extractor: Optional[ForwardArgumentExtractor]=None,
        n_classes: Optional[int]=None
    ) -> None:
        super().__init__(
            model,
            interpolate_mode,
            head_fusion_method,
            discard_ratio,
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
        rollout = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
        for grad, rel in zip(grads, rels):
            grad_x_rel = grad * rel
            attn_map = self.head_fusion_function(grad_x_rel)
            attn_map = self._discard(attn_map)
            identity = torch.eye(tgt_len).repeat(bsz, 1, 1).to(self.device)
            attn_map = .5 * attn_map + .5 * identity
            attn_map /= attn_map.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(rollout, attn_map)
        return rollout


class GenericAttention(AttentionRolloutBase):
    def __init__(
            self,
            model: Module,
            alpha: float=2.,
            beta: float=1.,
            stabilizer: float=1e-6,
            head_fusion_function: Optional[Callable[[Tensor], Tensor]]=None,
            n_classes: Optional[int]=None
        ) -> None:
        raise NotImplementedError