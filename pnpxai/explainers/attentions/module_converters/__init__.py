from importlib import util
from ...zennit.module_converter import module_converting_canonizer_factory

cfgs = dict()

if util.find_spec("timm"):
    import timm
    from .configs import TIMM_VIT_ATTENTION_CONVERTER_FACTORY_CONFIG
    cfgs[timm.models.vision_transformer.Attention] = TIMM_VIT_ATTENTION_CONVERTER_FACTORY_CONFIG

if util.find_spec("transformers"):
    import transformers
    from .configs import (
        TRANSFOMERS_BERT_ATTENTION_CONVERTER_FACTORY_CONFIG,
        TRANSFORMERS_VISUAL_BERT_ATTENTION_CONVERTER_FACTORY_CONFIG,
        TRANSFORMERS_VILT_LAYER_CONVERTER_FACTORY_CONFIG,
        TRANSFORMERS_LXMERT_SELF_ATTENTION_CONVERTER_FACTORY_CONFIG,
        TRANSFORMERS_LXMERT_CROSS_ATTENTION_CONVERTER_FACTORY_CONFIG,
    )
    cfgs[transformers.models.bert.modeling_bert.BertAttention] = TRANSFOMERS_BERT_ATTENTION_CONVERTER_FACTORY_CONFIG
    cfgs[transformers.models.visual_bert.modeling_visual_bert.VisualBertAttention] = TRANSFORMERS_VISUAL_BERT_ATTENTION_CONVERTER_FACTORY_CONFIG
    cfgs[transformers.models.vilt.modeling_vilt.ViltLayer] = TRANSFORMERS_VILT_LAYER_CONVERTER_FACTORY_CONFIG
    cfgs[transformers.models.lxmert.modeling_lxmert.LxmertSelfAttentionLayer] = TRANSFORMERS_LXMERT_SELF_ATTENTION_CONVERTER_FACTORY_CONFIG
    cfgs[transformers.models.lxmert.modeling_lxmert.LxmertCrossAttentionLayer] = TRANSFORMERS_LXMERT_CROSS_ATTENTION_CONVERTER_FACTORY_CONFIG

# import pdb; pdb.set_trace()

default_attention_converters = [
    module_converting_canonizer_factory(**cfg)()
    for cfg in cfgs.values()
] or None