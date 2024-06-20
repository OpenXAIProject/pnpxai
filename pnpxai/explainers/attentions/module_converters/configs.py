import torch
import timm
import transformers
from torch import nn

from .layers import StackAndSum
from .validations import *
from .warnings import *


def transform_attn_mask_of_transformers(in_args, in_kwargs, kept, out_module):
    # required: (tgt_len, src_len) or (bsz*num_heads, tgt_len, src_len)
    attn_mask = in_kwargs.get("attention_mask") or in_args[1]
    if attn_mask is None:
        return None
    bsz, num_heads, tgt_len, src_len = attn_mask.shape
    if bsz == 1 and num_heads == 1 and tgt_len == 1: # self attention
        attn_mask = attn_mask.repeat(1, 1, src_len, 1)
        return attn_mask.squeeze()
    elif bsz > 1 and num_heads == 1 and tgt_len == 1:
        attn_mask = attn_mask.repeat(
            1, out_module._converted_self_attention.num_heads,
            src_len, 1
        )
        return attn_mask.view(-1, src_len, src_len)
    else:
        raise Exception("Not covered yet")


# timm vit
TIMM_VIT_ATTENTION = {
    "out_module_type": nn.MultiheadAttention,
    "args": {
        "embed_dim": lambda in_module: in_module.head_dim * in_module.num_heads,
        "num_heads": lambda in_module: in_module.num_heads,
        "dropout": lambda in_module: in_module.attn_drop.p,
        "bias": lambda in_module: (in_module.qkv.bias.count_nonzero() > 0).item(),
        "add_bias_kv": False,
        "add_zero_attn": False,
        "kdim": None,
        "vdim": None,
        "batch_first": True,
        "device": lambda in_module: next(in_module.parameters()).device,
        "dtype": None,
    },
    "params": {
        "in_proj_weight": "qkv.weight",
        "out_proj.weight": "proj.weight",
        "in_proj_bias": "qkv.bias",
        "out_proj.bias": "proj.bias",
    },
    "forward": {
        "args": lambda in_args: tuple(in_args[0] for _ in range(3)),
        "kwargs": {
            "key_padding_mask": None,
            "need_weights": False,
            "attn_mask": None,
            "average_attn_weights": True,
            "is_causal": False,
        },
        "validations": [],
        "warnings": [],
    },
    "validations": [],
    "warnings": [],
}
TIMM_VIT_ATTENTION_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": timm.models.vision_transformer.Attention,
    "config_selector": True,
    "out_module_configs": {
        True: {
            "_converted_self_attention": TIMM_VIT_ATTENTION,
        },
    },
}

# transformers bert
TRANSFORMERS_BERT_ATTENTION = {
    "out_modules": {
        "_converted_self_attention": {
            "out_module_type": nn.MultiheadAttention,
            "args": {
                "embed_dim":  lambda in_module: in_module.self.attention_head_size * in_module.self.num_attention_heads,
                "num_heads": lambda in_module: in_module.self.num_attention_heads,
                "dropout": lambda in_module: in_module.self.dropout.p,
                "bias": True,
                "add_bias_kv": False,
                "add_zero_attn": False,
                "kdim": None,
                "vdim": None,
                "batch_first": True,
                "device": lambda in_module: next(in_module.self.parameters()).device,
                "dtype": None,
            },
            "params": {
                "in_proj_weight": lambda in_module_params: torch.cat([
                    in_module_params["self.query.weight"],
                    in_module_params["self.key.weight"],
                    in_module_params["self.value.weight"],
                ]),
                "in_proj_bias": lambda in_module_params: torch.cat([
                    in_module_params["self.query.bias"],
                    in_module_params["self.key.bias"],
                    in_module_params["self.value.bias"]
                ]),
                "out_proj.weight": "output.dense.weight",
                "out_proj.bias": "output.dense.bias",
            },
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: tuple(
                    in_args[0] for _ in range(3)
                ),
                "kwargs": {
                    "key_padding_mask": None,
                    "need_weights": False,
                    "attn_mask": transform_attn_mask_of_transformers,
                    "average_attn_weights": True,
                    "is_causal": False,
                },
                "keep_outputs": True,
            },
        },
        "_converted_dropout": {
            "module": lambda in_module: in_module.output.dropout,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_self_attention"][0],
                ),
                "keep_outputs": True,
            },
        },
        "_converted_layernorm": {
            "module": lambda in_module: in_module.output.LayerNorm,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_dropout"] + in_args[0],
                ),
                "keep_outputs": True,
            },
        },
    },
    "output_selector": lambda kept: (
        kept["outputs"]["_converted_layernorm"], # attention output
        kept["outputs"]["_converted_self_attention"][1], # attention weights
    ),
}
TRANSFOMERS_BERT_ATTENTION_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": transformers.models.bert.modeling_bert.BertAttention,
    "config_selector": lambda in_module: True,
    "out_module_configs": {True: TRANSFORMERS_BERT_ATTENTION},
}

# transformers visual bert
TRANSFORMER_VISUAL_BERT_ATTENTION = TRANSFORMERS_BERT_ATTENTION
TRANSFORMERS_VISUAL_BERT_ATTENTION_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": transformers.models.visual_bert.modeling_visual_bert.VisualBertAttention,
    "config_selector": lambda in_module: True,
    "out_module_configs": {
        True: TRANSFORMER_VISUAL_BERT_ATTENTION,
    },
}

# transformers vilt
TRANSFORMERS_VILT_ATTENTION = {
    "out_modules": {
        "_converted_self_attention": {
            "out_module_type": nn.MultiheadAttention,
            "args": {
                "embed_dim":  lambda in_module: in_module.attention.attention_head_size * in_module.attention.num_attention_heads,
                "num_heads": lambda in_module: in_module.attention.num_attention_heads,
                "dropout": lambda in_module: in_module.attention.dropout.p,
                "bias": True,
                "add_bias_kv": False,
                "add_zero_attn": False,
                "kdim": None,
                "vdim": None,
                "batch_first": True,
                "device": lambda in_module: next(in_module.attention.parameters()).device,
                "dtype": None,
            },
            "params": {
                "in_proj_weight": lambda in_module_params: torch.cat([
                    in_module_params["attention.query.weight"],
                    in_module_params["attention.key.weight"],
                    in_module_params["attention.value.weight"],
                ]),
                "in_proj_bias": lambda in_module_params: torch.cat([
                    in_module_params["attention.query.bias"],
                    in_module_params["attention.key.bias"],
                    in_module_params["attention.value.bias"]
                ]),
                "out_proj.weight": "output.dense.weight",
                "out_proj.bias": "output.dense.bias",
            },
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: tuple(
                    in_args[0] for _ in range(3)
                ),
                "kwargs": {
                    "key_padding_mask": None,
                    "need_weights": False,
                    "attn_mask": transform_attn_mask_of_transformers,
                    "average_attn_weights": True,
                    "is_causal": False,
                },
                "keep_outputs": True,
            },
        },
        "_converted_dropout": {
            "module": lambda in_module: in_module.output.dropout,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (kept["outputs"]["_converted_self_attention"][0],),
                "keep_outputs": True,
            },
        },
    },
    "output_selector": lambda kept: (
        kept["outputs"]["_converted_dropout"], # attention output
        kept["outputs"]["_converted_self_attention"][1], # attention weights
    ),
}
TRANSFORMERS_VILT_ATTENTION_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": transformers.models.vilt.modeling_vilt.ViltAttention,
    "config_selector": lambda in_module: True,
    "out_module_configs": {
        True: TRANSFORMERS_VILT_ATTENTION,
    }
}

TRANSFORMERS_VILT_OUTPUT = {
    "out_modules": {
        "_converted_dense": {
            "module": lambda in_module: in_module.dense,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (in_args[0],),
                "keep_outputs": True,
            },
        },
        "_converted_dropout": {
            "module": lambda in_module: in_module.dropout,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (kept["outputs"]["_converted_dense"],),
                "keep_outputs": True,
            },
        },
        "_converted_sum": {
            "module": lambda in_module: StackAndSum(),
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_dropout"],
                    in_args[1],
                ),
                "keep_outputs": True,
            },
        },
    },
    "output_selector": lambda kept: kept["outputs"]["_converted_sum"],
}
TRANSFORMERS_VILT_OUTPUT_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": transformers.models.vilt.modeling_vilt.ViltOutput,
    "config_selector": lambda in_module: True,
    "out_module_configs": {
        True: TRANSFORMERS_VILT_OUTPUT,
    }
}

TRANSFORMERS_VILT_LAYER = {
    "out_modules": {
        "_converted_layernorm_before": {
            "module": lambda in_module: in_module.layernorm_before,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (in_args[0],),
                "keep_outputs": True,
            },
        },
        "_converted_attention_attention": {
            "out_module_type": nn.MultiheadAttention,
            "args": {
                "embed_dim":  lambda in_module: in_module.attention.attention.attention_head_size * in_module.attention.attention.num_attention_heads,
                "num_heads": lambda in_module: in_module.attention.attention.num_attention_heads,
                "dropout": lambda in_module: in_module.attention.attention.dropout.p,
                "bias": True,
                "add_bias_kv": False,
                "add_zero_attn": False,
                "kdim": None,
                "vdim": None,
                "batch_first": True,
                "device": lambda in_module: next(in_module.attention.attention.parameters()).device,
                "dtype": None,
            },
            "params": {
                "in_proj_weight": lambda in_module_params: torch.cat([
                    in_module_params["attention.attention.query.weight"],
                    in_module_params["attention.attention.key.weight"],
                    in_module_params["attention.attention.value.weight"],
                ]),
                "in_proj_bias": lambda in_module_params: torch.cat([
                    in_module_params["attention.attention.query.bias"],
                    in_module_params["attention.attention.key.bias"],
                    in_module_params["attention.attention.value.bias"]
                ]),
                "out_proj.weight": "attention.output.dense.weight",
                "out_proj.bias": "attention.output.dense.bias",
            },
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: tuple(
                    kept["outputs"]["_converted_layernorm_before"] for _ in range(3)
                ),
                "kwargs": {
                    "key_padding_mask": None,
                    "need_weights": False,
                    "attn_mask": transform_attn_mask_of_transformers,
                    "average_attn_weights": True,
                    "is_causal": False,
                },
                "keep_outputs": True,
            },
        },
        "_converted_attention_output": {
            "module": lambda in_module: in_module.attention.output,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_attention_attention"][0],
                    kept["outputs"]["_converted_layernorm_before"],
                ),
                "keep_outputs": True,
            },
        },
        "_converted_first_residual_connection": {
            "module": lambda in_module: StackAndSum(),
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_attention_output"],
                    in_args[0],
                ),
                "keep_outputs": True,
            },
        },
        "_converted_layernorm_after": {
            "module": lambda in_module: in_module.layernorm_after,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_first_residual_connection"],
                ),
                "keep_outputs": True,
            },
        },
        "_converted_intermediate": {
            "module": lambda in_module: in_module.intermediate,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_layernorm_after"],
                ),
                "keep_outputs": True,
            },
        },
        "_converted_output_dense": {
            "module": lambda in_module: in_module.output.dense,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_intermediate"],
                ),
                "keep_outputs": True,
            },
        },
        "_converted_output_dropout": {
            "module": lambda in_module: in_module.output.dropout,
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_output_dense"],
                ),
                "keep_outputs": True,
            },
        },
        "_converted_second_residual_connection": {
            "module": lambda in_module: StackAndSum(),
            "forward": {
                "args": lambda in_args, in_kwargs, kept, out_module: (
                    kept["outputs"]["_converted_output_dropout"],
                    kept["outputs"]["_converted_first_residual_connection"],
                ),
                "keep_outputs": True,
            },
        },
    },
    "output_selector": lambda kept: (
        kept["outputs"]["_converted_second_residual_connection"],
        kept["outputs"]["_converted_attention_attention"][1],
    ),
}

TRANSFORMERS_VILT_LAYER_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": transformers.models.vilt.modeling_vilt.ViltLayer,
    "config_selector": lambda in_module: True,
    "out_module_configs": {
        True: TRANSFORMERS_VILT_LAYER,
    }
}

# transformers lxmert
TRANSFORMERS_LXMERT_SELF_ATTENTION = TRANSFORMERS_BERT_ATTENTION
TRANSFORMERS_LXMERT_SELF_ATTENTION_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": transformers.models.lxmert.modeling_lxmert.LxmertSelfAttentionLayer,
    "config_selector": lambda in_module: True,
    "out_module_configs": {
        True: TRANSFORMERS_LXMERT_SELF_ATTENTION,
    }
}

TRANSFORMERS_LXMERT_CROSS_ATTENTION = {
    "out_modules": {
        "_converted_cross_attention": {
            "out_module_type": nn.MultiheadAttention,
            "args": {
                "embed_dim":  lambda in_module: in_module.att.attention_head_size * in_module.att.num_attention_heads,
                "num_heads": lambda in_module: in_module.att.num_attention_heads,
                "dropout": lambda in_module: in_module.att.dropout.p,
                "bias": True,
                "add_bias_kv": False,
                "add_zero_attn": False,
                "kdim": None,
                "vdim": None,
                "batch_first": True,
                "device": lambda in_module: next(in_module.att.parameters()).device,
                "dtype": None,
            },
            "params": {
                "q_proj_weight": "att.query.weight",
                "k_proj_weight": "att.key.weight",
                "v_proj_weight": "att.value.weight",
                "in_proj_bias": lambda in_module_params: torch.cat([
                    in_module_params["att.query.bias"],
                    in_module_params["att.key.bias"],
                    in_module_params["att.value.bias"]
                ]),
                "out_proj.weight": "output.dense.weight",
                "out_proj.bias": "output.dense.bias",
            },
            "forward": {
                "args": lambda in_args, in_kwargs, kept: (in_args[0], in_args[1], in_args[1]),
                "kwargs": {
                    "key_padding_mask": None,
                    "need_weights": False,
                    "attn_mask": transform_attn_mask_of_transformers,
                    "average_attn_weights": True,
                    "is_causal": False,
                },
                "keep_outputs": True,
            },
        },
        "_converted_dropout": {
            "module": lambda in_module: in_module.output.dropout,
            "forward": {
                "args": lambda in_args, in_kwargs, kept: (kept["outputs"]["_converted_cross_attention"],),
                "keep_outputs": True,
            },
        },
        "_converted_layernorm": {
            "module": lambda in_module: in_module.output.LayerNorm,
            "forward": {
                "args": lambda in_args, in_kwargs, kept: (
                    kept["outputs"]["_converted_dropout"] + in_args[0],
                ),
                "keep_outputs": True,
            },
        },
    },
    "output_selector": lambda kept: (
        kept["outputs"]["_converted_layernorm"], # attention output
        kept["outputs"]["_converted_cross_attention"][1], # attention weights
    ),
}
TRANSFORMERS_LXMERT_CROSS_ATTENTION_CONVERTER_FACTORY_CONFIG = {
    "in_module_type": transformers.models.lxmert.modeling_lxmert.LxmertCrossAttentionLayer,
    "config_selector": lambda in_module: True,
    "out_module_configs": {
        True: TRANSFORMERS_LXMERT_CROSS_ATTENTION,
    }
}
