from torch.nn import MultiheadAttention
from torch.nn.modules import Module
from zennit.attribution import Attributor
from zennit.composites import LayerMapComposite

from pnpxai.explainers.attentions.rules import SavingAttention
from pnpxai.explainers.attentions.module_converters import default_attention_converters
from pnpxai.utils import format_into_tuple


class SavingAttentionAttributor(Attributor):
    def __init__(self, model: Module):
        layer_map = [(MultiheadAttention, SavingAttention())]
        composite = LayerMapComposite(
            layer_map=layer_map,
            canonizers=default_attention_converters,
        )
        super().__init__(model, composite, None)    
    
    def forward(self, input, attr_output_fn):
        input = format_into_tuple(input)
        _ = self.model(*input)
        attn_output_weights_all = [
            hook_ref.stored_tensors[hook_ref.saved_name]
            for hook_ref in self.composite.hook_refs
        ]
        return tuple(attn_output_weights_all)
