# setup
import os
import torch

# model
from torch.nn.modules import Module
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-imdb")

class Bert(BertForSequenceClassification):
    def forward(self, input_ids, token_type_ids, attention_mask):
        return super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).logits

model = Bert.from_pretrained('fabriceyhc/bert-base-uncased-imdb', num_labels=2)
model.eval()


# data
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import IMDB

class IMDBDataset(Dataset):
    def __init__(self, split='test'):
        data_iter = IMDB(split=split)
        self.annotations = [(line, label-1) for label, line in tqdm(data_iter)]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self.annotations[idx]

def collate_fn(batch):
    inputs = tokenizer(
        [d[0] for d in batch],
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    labels = torch.tensor([d[1] for d in batch])
    return tuple(inputs.values()), labels

dataset = IMDBDataset(split='test')
loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


# auto experiment
from pnpxai import AutoExperiment

input_extractor = lambda batch: batch[0]
label_extractor = lambda batch: batch[-1]
target_extractor = lambda outputs: outputs.argmax(-1)
forward_arg_extractor = lambda inputs: inputs[0] # input_ids
additional_forward_arg_extractor = lambda inputs: inputs[1:] # token_type_ids, attetion_mask
expr = AutoExperiment(
    model=model,
    layer=model.bert.embeddings.word_embeddings,
    data=loader,
    modality='text',
    question='why',
    evaluator_enabled=True,
    input_extractor=input_extractor,
    label_extractor=label_extractor,
    target_extractor=target_extractor,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
    target_labels=False,
)

expr.run(
    data_ids=[0, 42],
    explainer_ids=range(len(expr.all_explainers)),
    metrics_ids=range(2),
)


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from pnpxai.explainers.utils.postprocess.postprocess import postprocess_attr, normalize_relevance

span = lambda string, bg_color: f'''
    <span style='background-color: {bg_color};'>
        {string}
    </span>
'''

paragraph = lambda string: f'<p>{string}</p>'
bold = lambda string: f'<b>{string}</b>'


def plot_record(
    record,
    pooling_method='l2normsq',
    normalization_method='minmax',
    savepath='./test.html'
) -> str:
    label = model.config.id2label[record['label'].item()]
    pred = model.config.id2label[record['output'].argmax().item()]

    html = ''
    cmap = plt.get_cmap('YlGn')
    for expl in record['explanations']:
        line = ''
        line += bold(expl['explainer_nm'])
        if expl['explainer_nm'] not in ['Lime', 'KernelShap']:
            attr = postprocess_attr(expl['value'], channel_dim=-1)
        else:
            attr = normalize_relevance(expl['value'])
        for input_id, score in zip(record['input'][0], attr):
            word = tokenizer.decode(input_id).replace(' ', '')
            rgba = cmap(score.item())
            color = mcolors.rgb2hex(rgba)
            line += span(word, color)
        html += paragraph(line)
    with open(savepath, 'w') as f:
        f.write(html)
    return html

plot_record(expr.records[1], savepath='./auto_experiment_imdb_example.html')
