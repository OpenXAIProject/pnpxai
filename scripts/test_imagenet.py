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
from pnpxai.evaluator.metrics.base import Metric
from pnpxai.evaluator.metrics import AbPC, Complexity
from pnpxai.core.modality.modality import Modality

from tutorials.helpers import get_torchvision_model, get_imagenet_dataset


TORCHVISION_MODEL_CHOICES = [
    'resnet18',
    'vit_b_16',
]


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=TORCHVISION_MODEL_CHOICES, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1) # vit_b_16 x ig often raises memory error
parser.add_argument('--disable_gpu', action='store_true')
parser.add_argument('--fast_dev_run', action='store_true')

#------------------------------------------------------------------------------#
#---------------------------------- metrics -----------------------------------#
#------------------------------------------------------------------------------#

class CompoundMetric(Metric):
    def __init__(
        self,
        model,
        metrics,
        weights, 
        explainer=None,
        target_input_keys=None,
        additional_input_keys=None,
        output_modifier=None,
    ):
        super().__init__(
            model, explainer, target_input_keys,
            additional_input_keys, output_modifier,
        )
        assert len(metrics) == len(weights)
        self.metrics = metrics
        self.weights = weights

    def evaluate(self, inputs, targets, attrs):
        values = torch.zeros(attrs.size(0)).to(attrs.device)
        for weight, metric in zip(self.weights, self.metrics):
            values += weight * metric.set_explainer(self.explainer).evaluate(inputs, targets, attrs)
        return values

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

    # update metrics
    expr.metrics.delete('morf')
    expr.metrics.delete('lerf')

    expr.metrics.add('cmpx', Complexity)
    expr.metrics.add('cmpd', CompoundMetric)
    
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

            # setup for compound metrics
            metric_options = {}
            if metric_key == 'cmpd':
                metric_options['metrics'] = [
                    expr.create_metric('abpc'),
                    expr.create_metric('cmpx'),
                ]
                metric_options['weights'] = [.8, -.2]

            # fix n_samples for lime and kernel shap
            disable_tunable_params = {}
            if explainer_key in ['lime', 'kernel_shap']:
                disable_tunable_params['n_samples'] = 30

            # set direction
            direction = {
                'abpc': 'maximize',
                'cmpx': 'minimize',
                'cmpd': 'maximize',
            }.get(metric_key)

            # optimize
            opt_results = expr.optimize(
                explainer_key=explainer_key,
                metric_key=metric_key,
                metric_options=metric_options,
                direction=direction,
                disable_tunable_params=disable_tunable_params,
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

