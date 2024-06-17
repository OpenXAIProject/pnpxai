from torch import nn
from timm.models.vision_transformer import Attention as VisionTransformerAttention
from zennit.canonizers import AttributeCanonizer


class VisionTransformerAttentionCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)
    
    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, VisionTransformerAttention):
            attributes = {
                "forward": cls.forward.__get__(module),
                "canonizer_attention": convert_timm_attention_to_nn_multihead_attention(module),
            }
            return attributes
        return None
    
    @staticmethod
    def forward(self, x):
        out, _ = self.canonizer_attention(x, x, x, need_weights=False)
        return out


def convert_timm_attention_to_nn_multihead_attention(
        module: VisionTransformerAttention
    ) -> nn.MultiheadAttention:

    # validations
    assert isinstance(module.q_norm, nn.Identity)

    # kwargs for the new module
    bias = (module.qkv.bias.count_nonzero() > 0).item()
    converted_kwargs = {
        "embed_dim": module.head_dim * module.num_heads,
        "num_heads": module.num_heads,
        "dropout": module.attn_drop.p,
        "bias": bias,
        "add_bias_kv": False,
        "add_zero_attn": False,
        "batch_first": True,
    }

    # create the new module
    converted = nn.MultiheadAttention(**converted_kwargs)

    # move params to converted
    params = module.state_dict()
    converted_params = converted.state_dict()

    converted_params["in_proj_weight"] = params["qkv.weight"]
    converted_params["out_proj.weight"] = params["proj.weight"]

    if bias:
        converted_params["in_proj_bias"] = params["qkv.bias"]
        converted_params["out_proj.bias"] = params["proj.bias"]
    converted.load_state_dict(converted_params)
    return converted