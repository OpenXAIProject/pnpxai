'''
This script implements benchmark test on various explainers and gets the best
performing explainer on VQA task, using PnPXAI framework.

Prerequisites:

- Download VQA val dataset to your data_dir and unzip all. For example,
    
    ```bash
    mkdir vqa
    cd vqa
    wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
    wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
    wget https://visualqa.org/download.html
    unzip \*.zip
    ```
'''

import argparse
import os
import itertools
import inspect
import json
import glob
import random
import re
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

import captum
import torch
import omnixai
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import ViltForQuestionAnswering, ViltProcessor
from PIL import Image

from pnpxai import AutoExplanation
from pnpxai.core.modality.modality import Modality
from pnpxai.explainers.utils import PostProcessor
from pnpxai.explainers.utils.feature_masks import Felzenszwalb
from pnpxai.evaluator.metrics import Metric, Complexity


TARGET_MODEL_NAMES = ['vilt', 'visual_bert']
TARGET_METRIC_NAMES = ['mo_r_f', 'le_r_f', 'ab_p_c']


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=TARGET_MODEL_NAMES, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--obj_metric', type=str, choices=TARGET_METRIC_NAMES, default='ab_p_c')
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--disable_gpu', action='store_true')
parser.add_argument('--fast_dev_run', action='store_true')
parser.add_argument('--base_dir', type=str, default='.')
parser.add_argument('--explainers', type=str, default=None)


#------------------------------------------------------------------------------#
#----------------------------------- model ------------------------------------#
#------------------------------------------------------------------------------#

def load_pretrained_model(model_name: str):
    model_type, ckpt = {
        'vilt': (ViltForQuestionAnswering, 'dandelin/vilt-b32-finetuned-vqa'),
        'visual_bert': (None, None),
    }.get(model_name)
    return model_type.from_pretrained(ckpt)


#------------------------------------------------------------------------------#
#----------------------------------- data -------------------------------------#
#------------------------------------------------------------------------------#

def soft_encoding(unique_labels, label_to_id, annotations, fast_dev_run=False):
    def _get_score(count):
        return min(1., count / 3)

    if fast_dev_run:
        annotations = annotations[:10]
    for idx, annotation in tqdm(enumerate(annotations)):
        answers = annotation['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in unique_labels:
                continue
            labels.append(label_to_id[answer])
            scores.append(_get_score(answer_count[answer]))
        annotation['labels'] = labels
        annotation['scores'] = scores

        annotations[idx] = annotation

    return annotations


class VqaDataset(Dataset):
    """VQA (v2) dataset."""
    def __init__(self, data_dir, label2id, id2label, fast_dev_run=False):
        self.data_dir = data_dir
        self.label2id = label2id
        self.id2label = id2label

        self.unique_labels = list(label2id.keys())
        questions_dir = os.path.join(
            data_dir, 'v2_OpenEnded_mscoco_val2014_questions.json')
        annotations_dir = os.path.join(
            data_dir, 'v2_mscoco_val2014_annotations.json')
        images_dir = os.path.join(data_dir, 'val2014')

        with open(questions_dir) as f:
            self.questions = json.load(f)['questions']

        with open(annotations_dir) as f:
            self.annotations = json.load(f)['annotations']
            self.annotations = soft_encoding(
                unique_labels=self.unique_labels,
                label_to_id=self.label2id,
                annotations=self.annotations,
                fast_dev_run=fast_dev_run,
            )
            self.annotations = list(filter(
                lambda ann: len(ann['labels']) > 0,
                self.annotations)
            )

        self.images = glob.glob(os.path.join(images_dir, '*.jpg'))
        self.id_to_image = {
            int(os.path.basename(image).replace('.', '_').split('_')[-2]): image
            for image in self.images
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        question = self.questions[idx]['question']
        image = Image.open(self.id_to_image[annotation['image_id']]).convert('RGB')
        label_loc = np.argmax(annotation['scores'])
        label_id = annotation['labels'][label_loc]
        label = self.id2label[label_id]
        return {
            'image': image,
            'question': question,
            'answer': label
        }


class CollateSamplesFromVqaDataset:
    def __init__(self, processor, label2id):
        self.processor = processor
        self.label2id = label2id

    def __call__(self, batch):
        images = [d.pop('image') for d in batch]
        questions = [d.pop('question') for d in batch]
        answers = [d.pop('answer') for d in batch]
        batch = self.processor(
            images, questions,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        batch['labels'] = torch.tensor([
            self.label2id[ans] for ans in answers])
        return batch


#------------------------------------------------------------------------------#
#------------------------------ control group ---------------------------------#
#------------------------------------------------------------------------------#


# The other frameworks (omnixai, autoxai, xaitk, openxai) do not support vqa
CAPTUM_EXPLAINERS = {
    'grad_x_input': captum.attr.InputXGradient,
    'integrated_gradients': captum.attr.IntegratedGradients,
    'kernel_shap': captum.attr.KernelShap,
    'lime': captum.attr.Lime,
    # 'lrp_uniform_epsilon': captum.attr.LRP, # rule is not assigned to embedding layer
}


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


#------------------------------------------------------------------------------#
#---------------------------------- records ------------------------------------#
#------------------------------------------------------------------------------#


FIELDS = ['id', 'model', 'source', 'explainer', 'metric', 'value', 'params']


def get_last_index(jsonl_file):
    if not os.path.exists(jsonl_file):
        return -1  # No previous records

    with open(jsonl_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
        if lines:
            last_record = json.loads(lines[-1])  # Read last JSON object
            return last_record["id"]  # Get last recorded index
    return -1


def filter_records(jsonl_file, **filters):
    results = []
    if not os.path.exists(jsonl_file):
        return results
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if all(record.get(k) == v for k, v in filters.items() if v is not None):
                results.append(record)
    return results

def write_record(jsonl_file, record):
    with open(jsonl_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

#------------------------------------------------------------------------------#
#----------------------------------- main -------------------------------------#
#------------------------------------------------------------------------------#

def main(args):
    # setup
    use_gpu = torch.cuda.is_available() and not args.disable_gpu
    device = torch.device('cuda' if use_gpu else 'cpu')

    # prepare model
    model = load_pretrained_model(args.model)
    model.to(device)
    model.eval()

    # prepare dataloader
    dataset = VqaDataset(
        data_dir=args.data_dir,
        label2id=model.config.label2id,
        id2label=model.config.id2label,
        fast_dev_run=args.fast_dev_run,
    )
    if not args.fast_dev_run:
        random.seed(args.seed)
        sample_indices = random.sample(range(len(dataset)), args.n_samples)
        dataset = Subset(dataset, indices=sample_indices)
    collate_fn = CollateSamplesFromVqaDataset(
        processor={
            'vilt': ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa'),
            'visual_bert': None,
        }.get(args.model),
        label2id=model.config.label2id,
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
    )

    # prepare modality
    sample_batch = next(iter(dataloader))
    img_modality = Modality(
        dtype=sample_batch['pixel_values'].dtype,
        ndims=sample_batch['pixel_values'].dim(),
        pooling_dim=1,
    )


    #--------------------------------------------------------------------------#
    #--------------------------- auto explanation -----------------------------#
    #--------------------------------------------------------------------------#

    # create experiment
    expr = AutoExplanation(
        model=model,
        data=dataloader,
        modality=img_modality,
        target_input_keys=['pixel_values'], # img
        additional_input_keys=[
            'input_ids', # Do not target input_ids by regarding it as one of additional inputs
            'token_type_ids',
            'attention_mask',
            'pixel_mask',
        ],
        output_modifier=lambda outputs: outputs.logits,
        target_class_extractor=lambda modified_outputs: modified_outputs.argmax(-1),
        label_key='labels',
    )

    # update explainers
    target_explainer_keys = [] if args.explainers is None else args.explainers.split(',')
    for explainer_key in expr.explainers.choices:
        if explainer_key not in target_explainer_keys:
            expr.explainers.delete(explainer_key)

    # update metrics
    expr.metrics.delete('morf')
    expr.metrics.delete('lerf')

    expr.metrics.add('cmpx', Complexity)
    expr.metrics.add('cmpd', CompoundMetric)

    # update util functions
    for feature_mask_fn_key in expr.modality.util_functions['feature_mask_fn'].choices:
        if feature_mask_fn_key != 'checkerboard':
            expr.modality.util_functions['feature_mask_fn'].delete(feature_mask_fn_key)

    # set log file
    result_dir = os.path.join(args.base_dir, 'logs')
    os.makedirs(result_dir, exist_ok=True)
    record_file_name = os.path.basename(__file__).split('.')[0] + '.jsonl'
    if args.fast_dev_run:
        record_file_name = 'dev_' + record_file_name


    # optimize all
    best_params = defaultdict(dict)
    combs = list(itertools.product(
        expr.explainers.choices,
        expr.metrics.choices,
    ))
    pbar = tqdm(combs, total=len(combs))
    captum_explainers = {}
    for idx, (explainer_key, metric_key) in enumerate(pbar):
        if expr.is_tunable(explainer_key):
            results = filter_records(
                record_file_name,
                model=args.model,
                source='pnpxai',
                explainer=explainer_key,
                metric=metric_key,
            )
            if not results or args.fast_dev_run:
                pbar.set_description(f'[{idx}] Optimizing {explainer_key} on {metric_key}')
                metric_options = {}
                if metric_key == 'cmpd':
                    metric_options['metrics'] = [
                        expr.create_metric('abpc'),
                        expr.create_metric('cmpx'),
                    ]
                    metric_options['weights'] = [.8, -.2]
                disable_tunable_params = {}
                if explainer_key in ['lime', 'kernel_shap']:
                    disable_tunable_params['n_samples'] = 30
                opt_results = expr.optimize(
                    explainer_key=explainer_key,
                    metric_key=metric_key,
                    metric_options=metric_options,
                    direction={
                        'abpc': 'maximize',
                        'cmpx': 'minimize',
                        'cmpd': 'maximize',
                    }.get(metric_key),
                    disable_tunable_params=disable_tunable_params,
                    sampler='random',
                    seed=args.seed,
                    show_progress=not args.fast_dev_run,
                    n_trials=10 if args.fast_dev_run else 100,
                    num_threads=16,
                    errors='raise' if args.fast_dev_run else 'ignore',
                )
                write_record(
                    record_file_name,
                    {
                        'id': get_last_index(record_file_name)+1,
                        'model': args.model,
                        'source': 'pnpxai',
                        'explainer': explainer_key,
                        'metric': metric_key,
                        'value': opt_results.value,
                        'params': opt_results.params,
                    }
                )
            captum_explainer_cls = CAPTUM_EXPLAINERS.get(explainer_key)
            if captum_explainer_cls is not None:
                captum_explainers[explainer_key] = captum_explainer_cls(expr._wrapped_model)

    captum_evals = defaultdict(lambda: defaultdict(int))
    pp_default = PostProcessor(modality=img_modality)
    feature_mask_fn_default = Felzenszwalb()
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch in pbar:
        inputs = expr._wrapped_model.extract_inputs(batch) # dict input
        forward_args = tuple(expr._wrapped_model.extract_target_inputs(batch).values()) # tuple target inputs for captum
        additional_forward_args = tuple(expr._wrapped_model.extract_additional_inputs(batch).values()) # tuple additional inputs for captum
        outputs = expr._forward_batch(batch)
        targets = expr.target_class_extractor(outputs)
        for explainer_key, explainer in captum_explainers.items():
            attr_kwargs = {
                'inputs': forward_args[0],
                'target': targets,
                'additional_forward_args': additional_forward_args,
            }
            if 'feature_mask' in inspect.signature(explainer.attribute).parameters:
                attr_kwargs['feature_mask'] = feature_mask_fn_default(forward_args[0])
            attrs = explainer.attribute(**attr_kwargs)
            attrs_pp = pp_default(attrs)
            for metric_key in expr.metrics.choices:
                pbar.set_description(f'Evaluating captum {explainer_key} on {metric_key}')
                metric_options = {}
                if metric_key == 'cmpd':
                    metric_options['metrics'] = [
                        expr.create_metric('abpc'),
                        expr.create_metric('cmpx'),
                    ]
                    metric_options['weights'] = [.8, -.2]                
                metric = expr.create_metric(metric_key, **metric_options)
                evals_ = metric.set_explainer(explainer).evaluate(inputs, targets, attrs)
                captum_evals[explainer_key][metric_key] += evals_.sum().item()
    for explainer_key, evals in captum_evals.items():
        for metric_key, value in evals.items():
            write_record(
                record_file_name,
                {
                    'id': get_last_index(record_file_name)+1,
                    'model': args.model,
                    'source': 'captum',
                    'explainer': explainer_key,
                    'metric': metric_key,
                    'value': value / len(dataloader.dataset),
                    'params': None,
                }
            )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

