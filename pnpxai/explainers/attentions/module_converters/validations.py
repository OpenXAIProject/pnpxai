__all__ = [
    "_assert_position_embedding_type_is_absolute_for_bert_self_attention_of_transformers",
    "_assert_head_mask_is_none_for_bert_attention_of_transformers",
    "_assert_is_not_decoder_for_bert_self_attention_of_transformers",
]

def _assert_position_embedding_type_is_absolute_for_bert_self_attention_of_transformers(module):
    assert module.position_embedding_type == "absolute", f"module conversion for position_embedding_type='{module.position_embedding_type}' not supported."

def _assert_head_mask_is_none_for_bert_attention_of_transformers(args, kwargs):
    head_mask = kwargs["head_mask"] or args[2]
    assert head_mask is None

def _assert_is_not_decoder_for_bert_self_attention_of_transformers(module):
    assert not module.is_decoder
