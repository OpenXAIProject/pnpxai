import functools
import torch
from torch.utils.data import DataLoader
from pnpxai import AutoExplanation
from helpers import (
    get_vqa_dataset,
    get_vilt_model,
    get_vilt_processor,
    vilt_collate_fn,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_vilt_model('dandelin/vilt-b32-finetuned-vqa')
model = model.to(device)
model.eval()

dataset = get_vqa_dataset()
processor = get_vilt_processor('dandelin/vilt-b32-finetuned-vqa')
collate_fn = functools.partial(
    vilt_collate_fn,
    processor=processor,
    label2id=model.config.label2id,
)
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
)
input_extractor = lambda batch: tuple(d.to(device) for d in batch[:-1])
forward_arg_extractor = lambda inputs: inputs[:2]
additional_forward_arg_extractor = lambda inputs: inputs[2:]
label_extractor = lambda batch: batch[-1].to(device)
target_extractor = lambda outputs: outputs.argmax(-1).to(device)

expr = AutoExplanation(
    model=model,
    data=loader,
    layer=['pixel_values', model.vilt.embeddings.text_embeddings.word_embeddings],
    modality=('image', 'text'),
    input_extractor=input_extractor,
    label_extractor=label_extractor,
    target_extractor=target_extractor,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
    mask_token_id=processor.tokenizer.mask_token_id,
)


# test
for explainer_id in range(len(expr.manager.explainers)):
    optimized, objective, study = expr.optimize(
        data_id=0,
        explainer_id=explainer_id,
        metric_id=1,
        direction='maximize',
        sampler='tpe',
        n_trials=1,
    )
