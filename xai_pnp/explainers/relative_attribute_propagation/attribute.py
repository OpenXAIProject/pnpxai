from typing import Any, List

from plotly import express as px
from plotly import graph_objects as go
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

from xai_pnp.core._types import Model, DataSource
from xai_pnp.explainers._explainer import Explainer
from xai_pnp.explainers.relative_attribute_propagation.rap import RelativeAttributePropagation


class RAP(Explainer):
    def __init__(self, model: Model):
        super().__init__(model)
        self.method = RelativeAttributePropagation(model)

    def compute_pred(self, output):
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]

        T = pred.squeeze().cpu().numpy()
        T = np.expand_dims(T, 0)
        T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
        T = torch.from_numpy(T).type(torch.FloatTensor)
        Tt = Variable(T).cuda()
        return Tt

    def run(self, data: DataSource, *args: Any, **kwargs: Any) -> DataSource:
        attributions = []

        if torch.is_tensor(data):
            data = [data]

        device = next(self.model.parameters()).device

        for datum in data:
            if not (torch.is_tensor(datum)):
                datum = datum[0]

            datum = datum.to(device)
            outputs = self.model(datum)
            preds = self.compute_pred(outputs)
            relprop = self.method.relprop(preds, *args, **kwargs)
            relprop = relprop.sum(dim=1, keepdim=True)
            relprop = relprop.permute(0, 2, 3, 1)

            attributions.append(relprop)

        return torch.concat(attributions, dim=0)

    def format_outputs_for_visualization(self, outputs: Tensor) -> List[go.Figure]:
        return [px.imshow(output.sum(axis=-1)) for output in outputs]
