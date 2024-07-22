import torch
from torch.nn.modules import Module
from zennit.layer import Sum

class StackAndSum(Module):
    def __init__(self, dim=-1) -> None:
        super().__init__()
        self.dim = dim
        self.sum = Sum()
    
    def forward(self, x, y):
        out = torch.stack([x, y], dim=self.dim)
        out = self.sum(out)
        return out