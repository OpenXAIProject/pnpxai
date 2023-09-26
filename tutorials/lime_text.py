# ref: https://captum.ai/tutorials/Image_and_Text_Classification_LIME

from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from captum._utils.models.linear_model import SkLearnLasso

from xai_pnp import Project
from xai_pnp.explainers import LimeBase, Lime

##### User inputs

# build vocab
ag_ds = list(AG_NEWS(split='train'))
ag_train, ag_val = ag_ds[:100000], ag_ds[100000:]
tokenizer = get_tokenizer('basic_english')
word_counter = Counter()

for (label, line) in ag_train:
    word_counter.update(tokenizer(line))
unk_token = "<unk>"
voc = vocab(
    word_counter,
    min_freq = 10,
    specials=[unk_token]
)
default_index = -1
voc.set_default_index(-1)
voc.set_default_index(voc[unk_token])
num_class = len(set(label for label, _ in ag_train))


# model
class EmbeddingBagModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, num_class)

    def forward(self, inputs, offsets):
        embedded = self.embedding(inputs, offsets)
        return self.linear(embedded)


def collate_batch(batch):
    labels = torch.tensor([label - 1 for label, _ in batch]) 
    text_list = [tokenizer(line) for _, line in batch]
    
    # flatten tokens across the whole batch
    text = torch.tensor([voc[t] for tokens in text_list for t in tokens])
    # the offset of each example
    offsets = torch.tensor(
        [0] + [len(tokens) for tokens in text_list][:-1]
    ).cumsum(dim=0)

    return labels, text, offsets

EMB_SIZE = 64
model = EmbeddingBagModel(len(voc), EMB_SIZE, num_class)


# train
BATCH_SIZE = 64
EPOCHS = 1
train_loader = DataLoader(ag_train, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(ag_val, batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_batch)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(1, EPOCHS + 1):      
    # training
    model.train()
    total_acc, total_count = 0, 0
    
    for idx, (label, text, offsets) in enumerate(train_loader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss(predited_label, label).backward()
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if (idx + 1) % 500 == 0:
            print('epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(
                epoch, idx + 1, len(train_loader), total_acc / total_count
            ))
            total_acc, total_count = 0, 0       
    
    # evaluation
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for label, text, offsets in val_loader:
            predited_label = model(text, offsets)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    print('-' * 59)
    print('end of epoch {:3d} | valid accuracy {:8.3f} '.format(epoch, total_acc / total_count))
    print('-' * 59)


# inputs
test_label = 2  # {1: World, 2: Sports, 3: Business, 4: Sci/Tec}
test_line = ('US Men Have Right Touch in Relay Duel Against Australia THENS, Aug. 17 '
            '- So Michael Phelps is not going to match the seven gold medals won by Mark Spitz. '
            'And it is too early to tell if he will match Aleksandr Dityatin, '
            'the Soviet gymnast who won eight total medals in 1980.')

test_labels, test_text, test_offsets = collate_batch([(test_label, test_line)])


##### To be automated in the future

# setup for LIME
#   remove the batch dimension for the embedding-bag model
def forward_func(text, offsets):
    '''
    [GH] In some cases, forward function for explainer can be different from input model's one.
    In this case, the difference comes from a layer's property: nn.EmbeddingBag. I guess this
    is a potential challenge for automation.
    
    Please check comments in L191-200 and followings:

    https://captum.ai/tutorials/Image_and_Text_Classification_LIME > 2.3

    > ...
    > forward_func, the forward function of the model. Notice we cannot pass our model directly
    > since Captum always assumes the first dimension is batch while our embedding-bag requires
    > flattened indices. So we will add the dummy dimension later when calling attribute and
    > make a wrapper here to remove the dummy dimension before giving to our model. Please check
    > ...
    '''
    return model(text.squeeze(0), offsets)

#   encode text indices into latent representations & calculate cosine similarity
def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
    '''
    [GH] If a model includes an embedding layer, similarity function for captum LIME
    should be MANUALLY defined based on outputs of the embedding layer. This is also a
    potential challenge for automation.
    
    Please check comments in L202-224.
    '''
    original_emb = model.embedding(original_inp, None)
    perturbed_emb = model.embedding(perturbed_inp, None)
    distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
    return torch.exp(-1 * (distance ** 2) / 2)


#   binary vector where each word is selected independently and uniformly at random
def bernoulli_perturb(text, **kwargs):
    probs = torch.ones_like(text) * 0.5
    return torch.bernoulli(probs).long()

#   remove absenst token based on the intepretable representation sample
def interp_to_input(interp_sample, original_input, **kwargs):
    return original_input[interp_sample.bool()].view(original_input.size(0), -1)

# explain with cores
model.eval()
project = Project("test_project")
explainer = LimeBase(
    interpretable_model = SkLearnLasso(alpha=.01),
    similarity_func = exp_embedding_cosine_distance,
    perturb_func = bernoulli_perturb,
    perturb_interpretable_space = True,
    from_interp_rep_transform = interp_to_input,
    to_interp_rep_transform = None,
)
exp = project.make_experiment(forward_func, explainer) # [GH] model -> forward_func
exp.run(
    test_text.unsqueeze(0),
    target = test_labels,
    additional_forward_args = (test_offsets,),
    n_samples = 10000,
    show_progress = True
)

# [GH] Must provide different forward function from model's one: following raises error
#
# exp_m = project.make_experiment(model, explainer)
# exp_m.run(
#     test_text.unsqueeze(0),
#     target = test_labels,
#     additional_forward_args = (test_offsets,),
#     n_samples = 10000,
#     show_progress = True
# )

# [GH] Must use custom similarity_func: captum default lime runs without error but WRONG
#
# interpretable_model: SkLearnLasso(alpha=.01)
#   https://github.com/pytorch/captum/blob/master/captum/_utils/models/linear_model/model.py#L275
#
# similarity_func: exp cos
#   https://github.com/pytorch/captum/blob/master/captum/attr/_core/lime.py#L592
#   default similarity func convert the long tensor to float and get distance btw them (WRONG)
#
# perturb_func: bernoulli
#   https://github.com/pytorch/captum/blob/master/captum/attr/_core/lime.py#L641
#
# Following codes run without error but WRONG.
#
# explainer = Lime(similarity_func=exp_embedding_cosine_distance)
# exp_cd = project.make_experiment(forward_func, explainer)
# exp_cd.run(
#     test_text.unsqueeze(0),
#     target = test_labels,
#     additional_forward_args = (test_offsets,),
#     n_samples = 10000,
#     show_progress = True
# )


# visualization
attrs = exp.runs[0].outputs[0][0]

def show_text_attr(attrs) -> str:
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(tokenizer(test_line), attrs.tolist())
    ]
    return f"<p>{' '.join(token_marks)}</p>"

# simple server for vis
port = 9999
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(show_text_attr(attrs).encode('utf-8'))

httpd = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
print(f"Show result http://localhost:{port}")
httpd.serve_forever()