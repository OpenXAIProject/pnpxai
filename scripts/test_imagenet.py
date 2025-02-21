'''
This script implements benchmark test on various explainers and gets the best
performing explainer on ImageNet task, using PnPXAI framework.

Flags:
--fast_dev_run: runs the script with small samples and trials

Example:

```bash
python -m scripts.test_imagenet --model resnet18 --data_dir /path/to/pnpxai/tutorials/data/ImageNet --fast_dev_run
```
'''

import argparse
import itertools
from collections import defaultdict
from pprint import pprint

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from pnpxai import XaiRecommender, Experiment, AutoExplanation
from pnpxai.evaluator.metrics import AbPC
from pnpxai.core.modality.modality import Modality

from tutorials.helpers import get_torchvision_model, get_imagenet_dataset


TORCHVISION_MODEL_CHOICES = [
    'resnet18',
    'vit_b_16',
    # ...
]


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=TORCHVISION_MODEL_CHOICES, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1) # vit_b_16 x ig often raises memory error
parser.add_argument('--disable_gpu', action='store_true')
parser.add_argument('--fast_dev_run', action='store_true')


def main(args):
    # setup
    use_gpu = torch.cuda.is_available() and not args.disable_gpu
    device = torch.device('cuda' if use_gpu else 'cpu')

    # prepare model
    model, transform = get_torchvision_model(args.model)
    model.to(device)
    model.eval()

    # prepare data
    dataset = get_imagenet_dataset(
        transform,
        subset_size=10 if args.fast_dev_run else 1000,
        root_dir=args.data_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
    )

    # prepare modality
    sample_batch = next(iter(dataloader))
    modality = Modality(
        dtype=sample_batch[0].dtype,
        ndims=sample_batch[0].dim(),
        pooling_dim=1,
    )

    '''
    #--------------------------------------------------------------------------#
    #------------------------------- recommend --------------------------------#
    #--------------------------------------------------------------------------#

    # You can get pnpxai recommendation results without AutoExplanation as followings:

    recommended = XaiRecommender().recommend(
        modality=modality,
        model=model,
    )
    
    recommended.print_tabular()
    '''


    '''
    #--------------------------------------------------------------------------#
    #------------------------------ experiment --------------------------------#
    #--------------------------------------------------------------------------#

    # You can manually create experiment as followings:
    expr = Experiment(
        model=model,
        data=dataloader,
        modality=modality,
        target_input_keys=[0],  # feature location in batch from dataloader
        target_class_extractor=lambda outputs: outputs.argmax(-1),  # extract target classes from output batch
        label_key=-1,  # label location in batch from dataloader
    )

    # add recommended explainers recommended
    camel_to_snake = lambda name: re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    for explainer_type in recommended.explainers:
        name = camel_to_snake(explainer_type.__name__)
        expr.explainers.add(key=name, value=explainer_type)

    # add a metric
    expr.metrics.add(key='abpc', value=AbPC)
    '''


    #--------------------------------------------------------------------------#
    #--------------------------- auto explanation -----------------------------#
    #--------------------------------------------------------------------------#

    # create experiment using auto explanation
    expr = AutoExplanation(
        model=model,
        data=dataloader,
        modality=modality,
        target_input_keys=[0], # Current dataloader batches data as tuple of (inputs, targets). 0 means the location of inputs in the tuple
        target_class_extractor=lambda outputs: outputs.argmax(-1),
        label_key='labels',
        target_labels=False, # Gets attributions on the prediction for all explainer if False.
    )
    
    # You can browse available explainer_keys and metric_keys as followings:
    print(expr.explainers.choices)
    print(expr.metrics.choices)

    # optimize all
    records = []
    best_params = defaultdict(dict)
    combs = list(itertools.product(
        expr.explainers.choices,
        expr.metrics.choices,
    ))
    pbar = tqdm(combs, total=len(combs))
    for explainer_key, metric_key in pbar:
        if expr.is_tunable(explainer_key):  # skip if there's no tunable for an explainer
            pbar.set_description(f'Optimizing {explainer_key} on {metric_key}')
            direction = {
                'mo_r_f': 'minimize',
                'le_r_f': 'maximize',
                'ab_p_c': 'maximize',
            }.get(metric_key)
            opt_results = expr.optimize(
                explainer_key=explainer_key,
                metric_key=metric_key,
                direction=direction,
                sampler='random',
                num_threads=16,
                seed=42,
                show_progress=not args.fast_dev_run,
                n_trials=2 if args.fast_dev_run else 100,
            )
            records.append({
                'explainer': explainer_key,
                'metric': metric_key,
                'value': opt_results.study.best_trial.value,
            })
            best_params[explainer_key][metric_key] = opt_results.study.best_params
    df = pd.DataFrame.from_records(records)
    summary_table = df.set_index(
        ['explainer', 'metric'])['value'].unstack('metric')
    print('-------- Summary --------')
    print(summary_table)
    print('------ Best Params ------')
    pprint(best_params)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

