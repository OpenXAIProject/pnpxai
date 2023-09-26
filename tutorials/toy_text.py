import random
import re
import urllib
import urllib.request
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.io

from xai_pnp import Project
from xai_pnp.explainers import LayerIntegratedGradients


class ToyVocab:
    # https://baconipsum.com/json-api/
    BASE_URL = "https://baconipsum.com/api/"

    def __init__(self, query_params):
        self.query_params = query_params
        self._set_wordset()
    
    def _set_wordset(self):
        url = "?".join([
            self.BASE_URL,
            urllib.parse.urlencode(self.query_params)
        ])
        res = urllib.request.urlopen(url)
        lines = eval(res.readline())
        wordset = list()
        while lines:
            line = lines.pop()
            wordset.extend(re.findall("\w+", line.lower()))
        baseline = ["."]
        self.wordset = baseline + list(set(wordset))
        # res.readlines()[0]
    
    def __len__(self):
        return len(self.wordset)
    
    def stoi(self, s):
        return [self.wordset.index(_s) for _s in s]
    
    def itos(self, i):
        return [self.wordset[_i] for _i in i]
    
    def generate_random_sample(self, size: int=5):
        space = range(len(self))
        return [random.choice(space) for _ in range(size)]


vocab = ToyVocab({"type": "meat-and-filler"})


class TextClassificationModel(nn.Module):
    # https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 64)
        self.fc = nn.Linear(64, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, input):
        embedded = self.embedding(input)
        return self.fc(embedded)


model = TextClassificationModel().eval()

# defining model input tensor
input1 = torch.LongTensor(vocab.generate_random_sample(size=8))
input2 = torch.LongTensor(vocab.generate_random_sample(size=16))

# defining baselines for each input tensor
baseline1 = torch.LongTensor([0])
baseline2 = torch.LongTensor([1])

project = Project('test_project')
exp = project.make_experiment(model, LayerIntegratedGradients(layer=model.embedding))


# running in sequence: multiple inputs for a single run
exp.run([input1, input2], target=0, baselines=baseline1, method='gausslegendre')
exp.run([input1, input2], target=1, baselines=baseline1, method='gausslegendre')


# visualization: input1
bar1p = {
    "type": "bar",
    "x": vocab.itos(exp.runs[0].inputs[0]),
    "y": (exp.runs[1].outputs[0].sum(1) / torch.norm(exp.runs[0].outputs[0])).tolist(),
    "base": 0,
    "marker": {"color": "blue"},
    "name": "positive",
}

bar1n = {
    "type": "bar",
    "x": vocab.itos(exp.runs[1].inputs[0]),
    "y": (exp.runs[0].outputs[0].sum(1) / torch.norm(exp.runs[0].outputs[0])).tolist(),
    "base": 0,
    "marker": {"color": "red"},
    "name": "negative",
}


bar1 = [bar1p, bar1n]


# visualization: input2
bar2p = {
    "type": "bar",
    "x": vocab.itos(exp.runs[1].inputs[1]),
    "y": (exp.runs[1].outputs[1].sum(1) / torch.norm(exp.runs[1].outputs[1])).tolist(),
    "base": 0,
    "marker": {"color": "blue"},
    "name": "positive",
}

bar2n = {
    "type": "bar",
    "x": vocab.itos(exp.runs[0].inputs[1]),
    "y": (exp.runs[0].outputs[1].sum(1) / torch.norm(exp.runs[0].outputs[1])).tolist(),
    "base": 0,
    "marker": {"color": "red"},
    "name": "negative",
}


bar2 = [bar2p, bar2n]


fig = plotly.io.from_json(json.dumps(bar2))
fig.show()


# [GH] Running in sequence v.s. Running in batch
# From now, our `Explainer` is looping a sequence of inputs, not batching multiple inputs.
# Because almost of captum's explainers allows a batch of inputs and a batch of parameters,
# we can use them for batching "in the future".

# [GH] A related open question
# If a user is familiar with captum's interface, he/she might input a batch of inputs
# , a batch of targets, and a batch of baselines, to our `exp.run` method, for example,
# - exp.run(x_batch, target=tg_batch, baselines=bl_batch, method="gausslegendre")
# Then,
# - len(run.inputs) -> 1
# - len(run.outputs) -> 1
# while it contains *batch-size* number of inputs and outputs, for example,
# - run.inputs[0].shape[0] -> *batch-size*
# - run.outputs[0].shape[0] -> *batch-size*
# I guess
# - len(run.inputs) -> *batch-size*
# - len(run.outputs) -> *batch-size*
# will be more intuitive for users.


# Project
# - name: str
# - experiments: List[Experiment]
#   - Experiment
#       - model: Model
#       - explainer: Explainer
#       - runs: List[Run]
#           - Run
#               - model: Optional[Model]
#               - inputs: DataSource
#               - outputs: Optional[DataSource]
#               - ...

# how to visualize result is determined by `data`, `model`, and `explainer` -> `Experiment`
# `Run` is varying over `data` and parameters for `explainer` on a `Experiment`
# All the runs in an experment have same visualizations
# Is it possible 