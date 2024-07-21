from transformers import ViltForQuestionAnswering, ViltProcessor

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

model = Vilt.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')


from io import BytesIO
from PIL import Image
import requests
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
import torch
from torch.utils.data import Dataset, DataLoader

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

dataset = VQADataset()

def collate_fn(batch):
    imgs = [d[0] for d in batch]
    qsts = [d[1] for d in batch]
    inputs = processor(
        images=imgs,
        text=qsts,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    labels = torch.tensor([model.config.label2id[d[2]] for d in batch])
    return (
        inputs['pixel_values'],
        inputs['input_ids'],
        inputs['token_type_ids'],
        inputs['attention_mask'],
        inputs['pixel_mask'],
        labels,
    )

loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


from pnpxai import AutoExperiment

input_extractor = lambda batch: batch[:-1]
label_extractor = lambda batch: batch[-1]
target_extractor = lambda outputs: outputs.argmax(-1)

forward_arg_extractor = lambda inputs: inputs[:2]
additional_forward_arg_extractor = lambda inputs: inputs[2:]


expr = AutoExperiment(
    model=model,
    data=loader,
    layer=['pixel_values', model.vilt.embeddings.text_embeddings.word_embeddings],
    modality=('image', 'text'),
    question='why',
    evaluator_enabled=True,
    input_extractor=input_extractor,
    label_extractor=label_extractor,
    target_extractor=target_extractor,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
    mask_token_id=processor.tokenizer.mask_token_id,
)

expr.run(data_ids=[0], explainer_ids=range(len(expr.all_explainers)), metrics_ids=range(2))

from base64 import b64encode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from pnpxai.explainers.utils.postprocess import (
    relevance_pooling,
    normalize_relevance,
    postprocess_attr,
)
from helpers import denormalize_image

def plot_img_attr(
        attr,
        pooling_method='l2normsq',
        normalization_method='minmax',
        cmap='coolwarm',
    ):
    postprocessed = postprocess_attr(
        attr,
        channel_dim=0,
        pooling_method=pooling_method,
        normalization_method=normalization_method,
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    ax = ax.imshow(postprocessed.detach().numpy(), cmap=cmap)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # import pdb; pdb.set_trace()
    # ax.set_yticks([])
    # ax.set_xticks([])

    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    return f'<img src="data:image/png;base64, {b64encode(buffer.getvalue()).decode()}"/>'

# htmls
span = lambda string, bg_color: f'''
    <span style='background-color: {bg_color};'>
        {string}
    </span>
'''
paragraph = lambda string: f'<p>{string}</p>'
bold = lambda string: f'<b>{string}</b>'

def plot_qst_attr(
    qst,
    attr,
    pooling_method='l2normsq',
    normalization_method='minmax',
    do_pooling=True,
    cmap='YlGn'
) -> str:
    cmap = plt.get_cmap(cmap)
    html = ''
    if do_pooling:
        attr = relevance_pooling(attr, channel_dim=-1, method=pooling_method)
    attr = normalize_relevance(attr)
    for input_id, score in zip(qst, attr):
        word = processor.tokenizer.decode(input_id).replace(' ', '')
        rgba = cmap(score.item())
        color = mcolors.rgb2hex(rgba)
        html += span(word, color)
    return html

def plot_attrs(
        explainer_nm,
        attrs,
        qst,
        pooling_method=('l2normsq', 'l2normsq'),
        normalization_method=('minmax', 'minmax'),
        cmap=('coolwarm', 'YlGn'),
        do_qst_pooling=True,
    ):
    img_attr = plot_img_attr(
        attrs[0],
        pooling_method=pooling_method[0],
        normalization_method=normalization_method[0],
        cmap=cmap[0],
    )
    qst_attr = plot_qst_attr(
        qst,
        attrs[1],
        pooling_method=pooling_method[1],
        normalization_method=normalization_method[1],
        do_pooling=do_qst_pooling,
        cmap=cmap[1],
    )
    return f'''
    <div>
        <b>{explainer_nm}</b>
        <div class="row">
            <div class="column">{img_attr}</div>
            <div class="column">{qst_attr}</div>
        </div>
    </div>
    '''

CSS = '''
.row {
  display: flex;
}

.column {
  flex: 50%;
}
'''

def html(body):
    return f'''
    <!DOCTYPE html>
    <html>
      <head>
        <title>Photos</title>
        <style>{CSS}</style>
      </head>
      <body>
        {body}
      </body>
    </html>
    '''

def plot_record(record, savepath='./test.html'):
    body = ''
    for expl in record['explanations']:
        do_qst_pooling = expl['explainer_nm'] not in ['KernelShap', 'Lime']
        body += plot_attrs(
            expl['explainer_nm'],
            expl['value'],
            record['input'][1],
            pooling_method=('l2normsq', 'l2normsq'),
            normalization_method=('minmax','minmax'),
            do_qst_pooling=do_qst_pooling,
            cmap=('coolwarm', 'YlGn'),
        )
    with open(savepath, 'w') as f:
        f.write(html(body))
    return html

plot_record(expr.records[0], savepath='./auto_experiment_vqa_example.html')