import functools
import torch
from torch.utils.data import DataLoader
from pnpxai import AutoExplanationForTextClassification
from helpers import (
    get_imdb_dataset,
    get_bert_model,
    get_bert_tokenizer,
    bert_collate_fn,
)


# setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_bert_model(model_name='fabriceyhc/bert-base-uncased-imdb', num_labels=2)
model.to(device)

dataset = get_imdb_dataset(split='test')
tokenizer = get_bert_tokenizer(model_name='fabriceyhc/bert-base-uncased-imdb')
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=functools.partial(bert_collate_fn, tokenizer=tokenizer),
)
input_extractor = lambda batch: tuple(inp.to(device) for inp in batch[0])
forward_arg_extractor = lambda inputs: inputs[0]
additional_forward_arg_extractor = lambda inputs: inputs[1:]
label_extractor = lambda batch: batch[-1].to(device)
target_extractor = lambda outputs: outputs.argmax(-1).to(device)


# auto explanation
expr = AutoExplanationForTextClassification(
    model=model,
    data=loader,
    layer=model.bert.embeddings.word_embeddings,
    mask_token_id=tokenizer.mask_token_id,
    input_extractor=input_extractor,
    label_extractor=label_extractor,
    target_extractor=target_extractor,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)

# run
expr.run_batch(
    data_ids=[0, 42],
    explainer_id=0,
    postprocessor_id=0,
    metric_id=1,
)

# opt
for explainer_id in range(len(expr.manager.explainers)):
    optimized, objective, study = expr.optimize(
        data_id=0,
        explainer_id=0,
        metric_id=1,
    )