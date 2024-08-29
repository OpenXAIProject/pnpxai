import functools
import torch
from torch.utils.data import DataLoader
from pnpxai import AutoExplanationForVisualQuestionAnswering
from helpers import (
    get_vqa_dataset,
    get_vilt_model,
    get_vilt_processor,
    vilt_collate_fn,
)

torch.set_num_threads(1)
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

expr = AutoExplanationForVisualQuestionAnswering(
    model=model,
    data=loader,
    layer=['pixel_values', model.vilt.embeddings.text_embeddings.word_embeddings],
    mask_token_id=processor.tokenizer.mask_token_id,
    input_extractor=input_extractor,
    label_extractor=label_extractor,
    target_extractor=target_extractor,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)

expr.recommended.print_tabular()

optimized = expr.optimize(
    data_id=0,
    explainer_id=2,
    metric_id=1,
    direction='maximize', # less is better
    sampler='tpe', # Literal['tpe','random']
    n_trials=50, # by default, 50 for sampler in ['random', 'tpe'], None for ['grid']
    seed=42, # seed for sampler: by default, None
)

print('Best/Explainer:', optimized.explainer) # get the optimized explainer
print('Best/PostProcessor:', optimized.postprocessor) # get the optimized postprocessor
print('Best/value:', optimized.study.best_trial.value) # get the optimized value

# Every trial in study has its explainer and postprocessor in user attr.
i = 25
print(f'{i}th Trial/Explainer', optimized.study.trials[i].user_attrs['explainer']) # get the explainer of i-th trial
print(f'{i}th Trial/PostProcessor', optimized.study.trials[i].user_attrs['postprocessor']) # get the postprocessor of i-th trial
print(f'{i}th Trial/value', optimized.study.trials[i].value)

# For example, you can use optuna's API to get the explainer and postprocessor of the worst trial
def get_worst_trial(study):
    valid_trials = [trial for trial in study.trials if trial.value is not None]
    return sorted(valid_trials, key=lambda trial: trial.value)[0]

worst_trial = get_worst_trial(optimized.study)
print('Worst/Explainer:', worst_trial.user_attrs['explainer'])
print('Worst/PostProcessor', worst_trial.user_attrs['postprocessor'])
print('Worst/value', worst_trial.value)


# test
for explainer_id in range(len(expr.manager.explainers)):
    optimized = expr.optimize(
        data_id=0,
        explainer_id=explainer_id,
        metric_id=1,
        direction='maximize',
        sampler='tpe',
    )
