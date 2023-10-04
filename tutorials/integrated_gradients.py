from xai_pnp import Project
from xai_pnp.explainers import IntegratedGradients

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1):
        relu_out1 = F.relu(input1)
        return F.relu(relu_out1 - 1)


model = ToyModel()

# defining model input tensors
input1 = torch.tensor([3.0], requires_grad=True)
input2 = torch.tensor([1.0], requires_grad=True)

# defining baselines for each input tensor
baseline1 = torch.tensor([0.0])
baseline2 = torch.tensor([0.0])

project = Project('test_project')
exp = project.explain(IntegratedGradients(model))
exp.run(input1, baselines=baseline1, method='gausslegendre')

print(project.experiments[0].runs[0].outputs)
