from typing import Optional, List
import os
import json
import requests
import functools
from io import BytesIO
from pathlib import Path
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning

import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import ViltForQuestionAnswering, ViltProcessor

from tqdm import tqdm
from PIL import Image


# datasets

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'samples/')
        self.label_dir = os.path.join(self.root_dir, 'imagenet_class_index.json')
        
        with open(self.label_dir) as json_data:
            self.idx_to_labels = json.load(json_data)
        
        self.img_names = os.listdir(self.img_dir)
        self.img_names.sort()
        
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = idx
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def idx_to_label(self, idx):
        return self.idx_to_labels[str(idx)][1]

def get_imagenet_dataset(
        transform,
        subset_size: int=100, # ignored if indices is not None
        root_dir="./data/ImageNet",
        indices: Optional[List[int]]=None,
    ):
    os.chdir(Path(__file__).parent) # ensure path
    dataset = ImageNetDataset(root_dir=root_dir, transform=transform)
    if indices is not None:
        return Subset(dataset, indices=indices)
    indices = list(range(len(dataset)))
    subset = Subset(dataset, indices=indices[:subset_size])
    return subset


class IMDBDataset(Dataset):
    def __init__(self, split='test'):
        super().__init__()
        # data_iter = IMDB(split=split)
        # self.annotations = [(line, label-1) for label, line in tqdm(data_iter)]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations[idx]


def get_imdb_dataset(split='test'):
    return IMDBDataset(split=split)


disable_warnings(InsecureRequestWarning)

class VQADataset(Dataset):
    def __init__(self):
        super().__init__()
        res = requests.get('https://visualqa.org/balanced_data.json')
        self.annotations = eval(res.text)

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        data = self.annotations[idx]
        if isinstance(data['original_image'], str):
            print(f"Requesting {data['original_image']}...")
            res = requests.get(data['original_image'], verify=False)
            img = Image.open(BytesIO(res.content)).convert('RGB')
            data['original_image'] = img
        return data['original_image'], data['question'], data['original_answer']


def get_vqa_dataset():
    return VQADataset()



# models
def get_torchvision_model(model_name):
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = torchvision.models.get_model(model_name, weights=weights).eval()
    transform = weights.transforms()
    return model, transform


class Bert(BertForSequenceClassification):
    def forward(self, input_ids, token_type_ids, attention_mask):
        return super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).logits


def get_bert_model(model_name, num_labels):
    return Bert.from_pretrained(model_name, num_labels=num_labels)


class Vilt(ViltForQuestionAnswering):
    def forward(
        self,
        pixel_values,
        input_ids,
        token_type_ids,
        attention_mask,
        pixel_mask,
    ):
        return super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        ).logits


def get_vilt_model(model_name):
    return Vilt.from_pretrained(model_name)



# utils

img_to_np = lambda img: img.permute(1, 2, 0).detach().numpy()

def denormalize_image(inputs, mean, std):
    return img_to_np(
        inputs
        * Tensor(std)[:, None, None]
        + Tensor(mean)[:, None, None]
    )


def bert_collate_fn(batch, tokenizer=None):
    inputs = tokenizer(
        [d[0] for d in batch],
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    labels = torch.tensor([d[1] for d in batch])
    return tuple(inputs.values()), labels


def get_bert_tokenizer(model_name):
    return BertTokenizer.from_pretrained(model_name)


def get_vilt_processor(model_name):
    return ViltProcessor.from_pretrained(model_name)


def vilt_collate_fn(batch, processor=None, label2id=None):
    imgs = [d[0] for d in batch]
    qsts = [d[1] for d in batch]
    inputs = processor(
        images=imgs,
        text=qsts,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    labels = torch.tensor([label2id[d[2]] for d in batch])
    return (
        inputs['pixel_values'],
        inputs['input_ids'],
        inputs['token_type_ids'],
        inputs['attention_mask'],
        inputs['pixel_mask'],
        labels,
    )


def load_model_and_dataloader_for_tutorial(modality, device):
    if modality == 'image':
        model, transform = get_torchvision_model('resnet18')
        model = model.to(device)
        model.eval()
        dataset = get_imagenet_dataset(transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        return model, loader, transform
    elif modality == 'text':
        model = get_bert_model('fabriceyhc/bert-base-uncased-imdb', num_labels=2)
        model = model.to(device)
        model.eval()
        dataset = get_imdb_dataset(split='test')
        tokenizer = get_bert_tokenizer('fabriceyhc/bert-base-uncased-imdb')
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=functools.partial(bert_collate_fn, tokenizer=tokenizer)
        )
        return model, loader, tokenizer
    elif modality == ('image', 'text'):
        model = get_vilt_model('dandelin/vilt-b32-finetuned-vqa')
        model.to(device)
        model.eval()
        dataset = get_vqa_dataset()
        processor = get_vilt_processor('dandelin/vilt-b32-finetuned-vqa')
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=functools.partial(
                vilt_collate_fn,
                processor=processor,
                label2id=model.config.label2id,
            ),
        )
        return model, loader, processor
