#-----------------------------------------------------------------------------#
#----------------------------------- model -----------------------------------#
#-----------------------------------------------------------------------------#

from torch.nn.modules import Module
from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

class KlueBert(Module):
    def __init__(self):
        super().__init__()
        self.klue_bert = AutoModel.from_pretrained("klue/bert-base") # [GH] please load your model

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.klue_bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        ).pooler_output

model = KlueBert()

#-----------------------------------------------------------------------------#
#-------------------------------- recommender --------------------------------#
#-----------------------------------------------------------------------------#

from pnpxai import XaiRecommender

recommender = XaiRecommender()
recommended = recommender.recommend(modality='text', model=model)

'''
[GH]
The folowing line prints
    (i) the detected architecture,
    (ii) the recommended explaienrs, and
    (iii) the recommeded metrics
on your terminal.
'''

print('PNPXAI recommends the following explainers and metrics based on the detected model architecture:')
recommended.print_tabular()


#-----------------------------------------------------------------------------#
#---------------------------------- explain ----------------------------------#
#-----------------------------------------------------------------------------#

# setup explainer inputs
import torch

raw_text = "야 너희 립스틱 이쁘다 엄마가 성적 3등 올랐to다고 사준거야 나 오늘 남자친구랑 데이트 하는데 좀 빌려줘라 안 돼 저번에도 너 빌려 가서 망가트렸잖아 너 많이 컸다 이제 말대꾸도 하냐 그게 아니라 엄마가 백화점 가서 특별히 사 준 건데 그러니까 더 좋네 남자친구한테 잘 보이고 오늘 내가 빌려 갈테니까 그런 줄 알아 어디 가서 꼬바르지 말고 알아서 잘 해라 안 되는데"
tokenized = tokenizer(raw_text, return_tensors='pt')
inputs = tuple(tokenized.values()) # [GH] 'inputs' should be tensor or tuple of tensors in the new version of pnpxai
forward_arg_extractor = lambda inputs: inputs[0] # input_ids
additional_forward_arg_extractor = lambda inputs: inputs[1:] # (token_type_ids, attention_mask,)
labels = torch.tensor([2], dtype=torch.long) # true label
outputs = model(*inputs)
targets = outputs.argmax(-1)
layer_to_attribute = model.klue_bert.embeddings.word_embeddings


# setup explainers

# vanila gradient
from pnpxai.explainers import Gradient

grad = Gradient(
    model=model,
    layer=layer_to_attribute,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)


# grad x input
from pnpxai.explainers import GradientXInput

gradinp = GradientXInput(
    model=model,
    layer=layer_to_attribute,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)


# smooth grad
from pnpxai.explainers import SmoothGrad

sg = SmoothGrad(
    model=model,
    layer=layer_to_attribute,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)


# var grad
from pnpxai.explainers import VarGrad

vg = VarGrad(
    model=model,
    layer=layer_to_attribute,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)


# integrated gradients
from pnpxai.explainers import IntegratedGradients

ig = IntegratedGradients(
    model=model,
    baseline_fn=lambda input_ids: input_ids * 0, # pad token
    layer=layer_to_attribute,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)

# lrp uniform epsilon
from pnpxai.explainers import LRPUniformEpsilon

lrp = LRPUniformEpsilon(
    model=model,
    layer=layer_to_attribute,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
    epsilon=.25,
)


# attention rollout
from pnpxai.explainers import AttentionRollout

ar = AttentionRollout(
    model=model,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)


# transformer attribution
from pnpxai.explainers import TransformerAttribution

ta = TransformerAttribution(
    model=model,
    layer=layer_to_attribute,
    forward_arg_extractor=forward_arg_extractor,
    additional_forward_arg_extractor=additional_forward_arg_extractor,
)



explainers = [grad, gradinp, sg, vg, ig, lrp, ar, ta]


# explain
records = []
for explainer in explainers:
    attrs = explainer.attribute(inputs=inputs, targets=targets)
    print(explainer.__class__.__name__, attrs.shape)
    records.append({
        'explainer_nm': explainer.__class__.__name__,
        'inputs': inputs,
        'attrs': attrs,
    })


#-----------------------------------------------------------------------------#
#--------------------------------- visualize ---------------------------------#
#-----------------------------------------------------------------------------#

import imgkit # [GH] please ensure wkhtmltopdf and xvfb are installed in your os: sudo apt-get install wkhtmltopdf xvfb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pnpxai.explainers.postprocess import relevance_pooling


span = lambda string, bg_color: f'''
    <span style='background-color: {bg_color}; font-size: 50px;'>
        {string}
    </span>
'''

def plot_attr(
    input_ids,
    text_attr,
    skip_cls_token=False,
    skip_sep_token=False,
    skip_pad_token=True,
) -> str:
    cmap = plt.get_cmap('YlGn')
    html = span("<b>Attributions</b>: ", "null")
    for input_id, score in zip(input_ids, text_attr):
        if skip_cls_token and input_id == tokenizer.cls_token_id:
            continue
        if skip_sep_token and input_id == tokenizer.sep_token_id:
            continue
        if skip_pad_token and input_id == tokenizer.pad_token_id:
            continue
        word = tokenizer.decode(input_id).replace(" ","")
        rgba = cmap(score.item())
        color = mcolors.rgb2hex(rgba)
        html += span(word, color)
        html += span(" ", "null")
    return html


for record in records:
    print(record['explainer_nm'])
    inp = record['inputs'][0][0]
    attr = record['attrs'][0]
    attr = relevance_pooling(attr, channel_dim=-1, method='sumpos')
    attr = (attr - attr.min()) / (attr.max() - attr.min())
    html = plot_attr(inp, attr)
    imgkit.from_string(html, f"./sogang_example_{record['explainer_nm']}.jpg", options={"xvfb":""})
