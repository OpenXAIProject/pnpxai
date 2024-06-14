from .BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from .BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import AutoTokenizer

from pnpxai.explainers import Explainer
from pnpxai.core._types import Model, DataSource, Task, Tensor

import torch


class TransformerAttention(Explainer):
    def __init__(self, model: BertForSequenceClassification):
        super().__init__(model=model)
        self.source = Generator(model)

    def attribute(
            self,
            inputs: DataSource,
            targets: DataSource,
            tokenizer: AutoTokenizer
    ):
        encoding = tokenizer(inputs, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        pred_result = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        pred_label = torch.argmax(pred_result, dim=1).item()

        pred_softmax = torch.nn.functional.softmax(pred_result, dim=-1)[0]
        pred_prob = pred_softmax[pred_label].item()

        # generate an explanation for the input
        expl = self.source.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
        # normalize scores
        expl = (expl - expl.min()) / (expl.max() - expl.min())

        tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())

        return expl, tokens, pred_label, pred_prob
