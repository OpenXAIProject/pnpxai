import time
import torch
import torchvision.transforms.functional as TF
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
from pnpxai import AutoExplanationForImageClassification

from helpers import get_imagenet_dataset, get_torchvision_model, denormalize_image

import torch.nn as nn
import torch.nn.functional as F
import torchvision

#------------------------------------------------------------------------------#
#-------------------------------- basic usage ---------------------------------#
#------------------------------------------------------------------------------#

# setup
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, transform = get_torchvision_model('vit_b_16')
dataset = get_imagenet_dataset(transform, indices=range(1000))
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# create auto explanation
expr = AutoExplanationForImageClassification(
    model=model.to(device),
    data=loader,
    input_extractor=lambda batch: batch[0].to(device),
    label_extractor=lambda batch: batch[-1].to(device),
    target_extractor=lambda outputs: outputs.argmax(-1).to(device),
    target_labels=False, # target prediction if False
)


# browse the recommended
expr.recommended.print_tabular() # recommendation
expr.recommended.explainers # -> List[Type[Explainer]]

# browse explainers and metrics
expr.manager.explainers # -> List[Explainer]
expr.manager.metrics # -> List[Metric]

expr.manager.get_explainer_by_id(7) # -> Explainer. In this case, LRPEpsilonGammaBox
expr.manager.get_postprocessor_by_id(0) # -> PostProcessor. In this case, PostProcessor(pooling_method='sumpos', normalization_method='minmax')
expr.manager.get_metric_by_id(0) # -> Metric. In this case, AbPC

# explain and evaluate

results = expr.run_batch(
    data_ids=range(4),
    explainer_id=0,
    postprocessor_id=0,
    metric_id=0,
)



#------------------------------------------------------------------------------#
#------------------------------- optimization ---------------------------------#
#------------------------------------------------------------------------------#

# user inputs
explainer_id = 5 # explainer_id to be optimized: KernelShap
metric_id = 1 # metric_id to be used as objective: AbPC
data_id = 2

# optimize: returns optimal explainer id, optimal postprocessor id, (and study)
optimized = expr.optimize(
    data_id=data_id,
    explainer_id=explainer_id,
    metric_id=metric_id,
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


# # test
# for explainer_id in range(len(expr.manager.explainers)):
#     # optimize: returns optimal explainer id, optimal postprocessor id, (and study)
#     optimized = expr.optimize(
#         data_id=data_id,
#         explainer_id=explainer_id,
#         metric_id=metric_id,
#         direction='maximize', # larger better
#         sampler='tpe', # Literal['tpe','random']
#         n_trials=50, # by default, 50 for sampler in ['random', 'tpe'], None for ['grid']
#         seed=42, # seed for sampler: by default, None
#     )
