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
import json
import glob
import re
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import ViltForQuestionAnswering, ViltProcessor
from PIL import Image

from pnpxai import (
    XaiRecommender,
    AutoExplanation,
    Experiment,
)
from pnpxai.evaluator.metrics import AbPC
from pnpxai.core.modality.modality import Modality


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['vilt', 'visual_bert'], required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--disable_gpu', action='store_true')
parser.add_argument('--fast_dev_run', action='store_true')


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
        annotations = annotations[:100]
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
        image = Image.open(self.id_to_image[annotation['image_id']])
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
    if args.fast_dev_run:
        args.num_workers = 0
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
    qst_modality = Modality(
        dtype=sample_batch['input_ids'].dtype,
        ndims=sample_batch['input_ids'].dim(),
        pooling_dim=-1,
        mask_token_id=0,
    )

    '''
    #--------------------------------------------------------------------------#
    #------------------------------- recommend --------------------------------#
    #--------------------------------------------------------------------------#

    # You can get pnpxai recommendation results without AutoExplanation as followings:

    recommended = XaiRecommender().recommend(
        modality=(img_modality, qst_modality),
        model=model,
    )
    
    recommended.print_tabular()
    '''


    '''
    #--------------------------------------------------------------------------#
    #------------------------------ experiment --------------------------------#
    #--------------------------------------------------------------------------#

    # You can manually create experiment as followings:
    # expr = Experiment(
        model=model,
        data=dataloader,
        modality=(img_modality, qst_modality), # ensure the order to follow target_input_keys
        target_input_keys=['pixel_values', 'input_ids'], # img, qst
        additional_input_keys=[
            'token_type_ids',
            'attention_mask',
            'pixel_mask',
        ],
        output_modifier=lambda outputs: outputs.logits,
        target_class_extractor=lambda modified_outputs: modified_outputs.argmax(-1),
        label_key='labels',
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
        modality=(img_modality, qst_modality), # ensure the order to follow target_input_keys
        target_layer=['pixel_values', model.vilt.embeddings.text_embeddings],
        target_input_keys=['pixel_values', 'input_ids'], # img, qst
        additional_input_keys=[
            'token_type_ids',
            'attention_mask',
            'pixel_mask',
        ],
        output_modifier=lambda outputs: outputs.logits,
        target_class_extractor=lambda modified_outputs: modified_outputs.argmax(-1),
        label_key='labels',
    )

    # optimize all
    records = []
    best_params = defaultdict(dict)
    combs = list(itertools.product(
        expr.explainers.choices,
        expr.metrics.choices,
    ))
    pbar = tqdm(combs, total=len(combs))
    for explainer_key, metric_key in pbar:
        if expr.is_tunable(explainer_key):
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
                seed=42,
                show_progress=not args.fast_dev_run,
                n_trials=2 if args.fast_dev_run else 100,
                num_threads=16,
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

