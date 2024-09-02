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
# device = torch.device('cpu')
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

optimized = expr.optimize(
    data_id=0,
    explainer_id=5,
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
